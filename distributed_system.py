import torch


def tensor_copy(x:torch.Tensor,
                is_new_param=True,
                copy_grad=False) -> torch.tensor:
    """
    将x的相应信息复制到y中 \n
    | is_new_param | copy_grad | 效果 \n
    |     True     |   False   | y是值与x相同的新参数, 初始梯度=None \n
    |     True     |   True    | y是值与x相同, 梯度也与x相同的新参数 \n
    |     False    |   True    | y = x \n
    |     False    |   False   | y是值与x相同的常值tensor
    """
    if(is_new_param and not copy_grad):
        y = x.clone().detach()
        y.requires_grad = True
        return y
    if(is_new_param and copy_grad):
        y = x.clone().detach()
        y.requires_grad = True
        if(x.grad != None):
            y.grad = x.grad.clone()
        return y
    if(not is_new_param and copy_grad):
        y = x
        return y
    if(not is_new_param and not copy_grad):
        y = x.clone().detach()
        return y


class node():
    """分布式系统节点的类"""
    def __init__(self, func,
                 data:list[torch.Tensor],
                 params:torch.Tensor,
                 mailbox:list[torch.Tensor]=[]) -> None:
        """
        节点的结构:
        - node.{func = func[i], data = data[i], params = params
                mailbox = [data_to_communicate]}
        """
        # 初始化本地函数和参数
        self.func = func
        self.params = tensor_copy(params)
        self.mailbox = mailbox

        # 初始化数据
        self.data_num = len(data)
        self.data_dim = data[0].shape[0]
        self.data = torch.zeros((self.data_num, self.data_dim), requires_grad=False)
        for i in range(0, self.data_num):
            # 将数据转化为Tensor矩阵, 第i行代表第i组数据
            self.data[i] = data[i]
        

    def forward(self, requires_grad=True) -> torch.Tensor:
        """
        计算该节点的函数值
        """
        if(requires_grad):
            y = self.func(self.data, self.params)
        else:
            with torch.no_grad():
                y = self.func(self.data, self.params)
        return y
    
    
    def backward(self) -> None:
        """
        反向传播, 求梯度
        """
        if(self.params.grad != None):
            self.params.grad.zero_()
        y = self.forward()
        y.backward()


    def backward_mailbox(self) -> None:
        """
        对通信区域中的参数求梯度
        """
        tmp = tensor_copy(self.params, True, True)
        for others_params in self.mailbox:
            self.params = others_params
            self.backward()
        self.params = tensor_copy(tmp, True, True)


    def iterate(self, grad_func) -> None:
        """
        梯度下降迭代参数 \n
        grad_func = grad_func(_params_local:Tensor,\n
        _params_mailbox:list[Tensor])\n
        x_i(k+1) = x_i(k) - g(x_i(k), x_j(k))
        """
        with torch.no_grad():
            self.params -= grad_func(self.params, self.mailbox)


class distributed_sys(list[node]):
    """分布式系统的类"""
    def __init__(self, func,
                 data:list[list[torch.Tensor]],
                 params:torch.Tensor) -> None:
        """
        初始化分布式系统, 生成一个具有若干节点的列表
        - func: 每个节点的目标函数, y[i] = func[i](data[i], params)
        - data: 每个节点的数据 : [[第i个节点的数据:Tensor]]
        - params: 待优化参数
        """
        # 节点总信息
        self.v_num = len(data)
        self.params = tensor_copy(params)
        self.data = data
        self.com_mat = torch.tensor([])
        # 接受函数列表, 也接受单个函数的输入
        if(type(func) == list):
            self.func = func
        else:
            self.func = [func for _ in range(0, self.v_num)]

        # 初始化节点
        for i in range(0, self.v_num):
            node_i = node(self.func[i], data[i], self.params)
            self.append(node_i)


    def create_com_mat(self,
                       disconnection:list[tuple]=[],
                       connection:list[tuple]=[],
                       is_fullconnected=False) -> None:
        """
        创建通信矩阵, 默认节点环形相连(1~2~3~...~m~1)
        - disconnection: 给出了哪些节点不相连: [(i, j)]表示i, j不能相互通信
        - connection: 给出了哪些节点相连: [(i, j)]表示i, j可以相互通信
        - is_fullconnected: 将所有节点相连, 该设置优先级最高
        """
        if(is_fullconnected or disconnection):
            # 设置所有节点相连, 或给出了哪些节点不相连
            # 连接所有节点
            self.com_mat = torch.tensor(
                [[1] * self.v_num for _ in range(0, self.v_num)],
                dtype=torch.int)
            for i in range(0, self.v_num):
                self.com_mat[i][i] = 0

            if(is_fullconnected):
                # 要求所有节点相连
                pass
            else:
                # 切断不相连的节点
                for each in disconnection:
                    i, j = each
                    self.com_mat[i][j] = 0
                    self.com_mat[j][i] = 0
        else:
            # 将节点环形相连
            self.com_mat = torch.tensor(
                [[0] * self.v_num for _ in range(0, self.v_num)],
                dtype=torch.int)
            for i in range(0, self.v_num - 1):
                self.com_mat[i][i + 1] = 1
                self.com_mat[i + 1][i] = 1
            i += 1
            self.com_mat[i][0] = 1
            self.com_mat[0][i] = 1

            if(connection):
                # 给出了哪些节点相连
                for each in connection:
                    i, j = each
                    self.com_mat[i][j] = 1
                    self.com_mat[j][i] = 1


    def connect_nodes(self, i:int, j:int) -> None:
        """连接两个节点"""
        if(self.com_mat.shape[0] == 0):
            # 未创建通信矩阵
            self.create_com_mat()

        if(i != j):
            self.com_mat[i][j] = 1
            self.com_mat[j][i] = 1


    def disconnect_nodes(self, i:int, j:int) -> None:
        """切断两个节点"""
        if(self.com_mat.shape[0] == 0):
            # 未创建通信矩阵
            self.create_com_mat()

        if(i != j):
            self.com_mat[i][j] = 0
            self.com_mat[j][i] = 0


    def forward_all(self, requires_grad=True, is_distributed=True) -> torch.Tensor:
        """
        计算所有节点的函数值
        - requires_grad: 是否需要计算梯度
        - is_distributed=True: 默认分布式计算, 每个节点更新本地参数
        - is_distributed=False: 集中计算, 更新总参数
        """
        val = torch.tensor([0.0])
        if(is_distributed):
            for each in self:
                val += each.forward(requires_grad)
        else:
            if(requires_grad):
                for each in self:
                    val += each.func(each.data, self.params)
            else:
                with torch.no_grad():
                    for each in self:
                        val += each.func(each.data, self.params)
        return val

                
    def backward_all(self, is_distributed=True) -> None:
        """
        所有节点反向传播求梯度
        - is_distributed=True: 默认分布式计算, 每个节点更新本地参数
        - is_distributed=False: 集中计算, 更新总参数
        """
        if(is_distributed):
            for node_i in self:
                node_i.backward()
        else:
            if(self.params.grad != None):
                self.params.grad.zero_()
            val = self.forward_all()
            val.backward()


    def backward_mailbox_all(self) -> None:
        """
        所有节点对通信区域中的参数求梯度
        """
        for node_i in self:
            node_i.backward_mailbox()


    def send(self) -> None:
        """
        每个节点给相邻节点发送参数
        """
        # 清空所有节点的通信区域
        for node_i in self:
            node_i.mailbox.clear()

        index = torch.arange(self.v_num, dtype=torch.int)
        # vi节点给vj节点发信
        for vi in range(0, self.v_num):
            node_i = self[vi]

            # 搜索周围节点
            neighbors = [self[each.item()]
                for each in index[self.com_mat[vi] == 1]]
            
            # 给vj节点发信
            for node_j in neighbors:
                node_j.mailbox.append(tensor_copy(node_i.params))


    def getback(self) -> None:
        """
        每个节点取回发送出去的参数
        """
        index = torch.arange(self.v_num, dtype=torch.int)
        # vi节点接收vi节点的信息
        for vi in range(0, self.v_num):
            node_i = self[vi]

            # 搜索周围节点
            neighbors = [self[each.item()]
                for each in index[self.com_mat[vi] == 1]]
            
            # 接收vi节点的信息
            for node_j in neighbors:
                node_i.mailbox.append(node_j.mailbox.pop(0))


    def iterate_all(self, grad_func) -> None:
        """
        每个节点做梯度下降迭代
        """
        for node_i in self:
            node_i.iterate(grad_func)


    def renew_params(self, params:torch.Tensor) -> torch.Tensor:
        """
        将新的参数params分发给每个节点
        """
        for node_i in self:
            node_i.params = tensor_copy(params)

