#created by WANG HUAN
import numpy as np

class HMM(object):
    def __init__(self, A, B, pi):
        '''
        A : 转移矩阵 [N,N]
        B : 生成概率 [N,2]
        pi : 初始概率分布 [N,1]
        以上三个值组成了HMM的参数lamda
        '''
        self.A = np.array(A)
        self.B = np.array(B)
        self.pi = np.array(pi)
    

    def forward(self, O):
        '''
        O : 观察序列 [M,1]
        求解P(O|lamda)
        '''
        num_states = np.shape(self.B)[0]
        T = np.shape(O)[0]

        alpha = np.zeros((num_states,T)) # 存储计算过程
        alpha[:,0] = self.B[:,O[0]]*self.pi

        for t in range(1,T):
            for s in range(num_states):
                alpha[s,t] = self.B[s,O[t]] * np.sum(alpha[:,t-1] * self.A[:,s])
        return alpha,np.sum(alpha[:,-1])
    
    def viterbi(self, O):
        '''
        O : 观察序列 [M,1]
        求解最可能的状态序列
        '''
        num_states = np.shape(self.B)[0]
        T = np.shape(O)[0]

        delta = np.zeros((num_states,T)) # 存储计算过程
        delta[:,0] = self.B[:,O[0]]*self.pi

        for t in range(1,T):
            for s in range(num_states):
                delta[s,t] = self.B[s,O[t]] * np.max(delta[:,t-1] * self.A[:,s])
        path = np.argmax(delta, 0)+1
        return delta, path



if __name__ == "__main__":
    A = [[0.4, 0.6, 0],[0, 0.5, 0.5], [0, 0, 0.9]]
    B = [[0.2, 0.8], [0.6, 0.4], [0.4, 0.6]]
    pi = [1, 0, 0]
    O = [0,1,1,0] # 表示ABAB
    hmm = HMM(A, B, pi)
    alpha, P = hmm.forward(O)
    print("P(O|lamda):{:.5}".format(P))
    print("alpha matrix:", alpha)
    delta, path = hmm.viterbi(O)
    print("path:",path) #代表状态的下标
    print("delta matrix", delta)
