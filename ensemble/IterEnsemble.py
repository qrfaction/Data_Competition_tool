import heapq

class history_score(object):
    def __init__(self,score,pred,info=None):
        self.score = score       #评估分数
        self.pred = pred         #输出结果
        self.info = info         #其他附带信息
    def __lt__(self,other):
        return self.score < other.score

    def __str__(self):
        return "score:"+str(self.score)


class iterEnsemble(object):

    def __init__(self,N):
        self.history = []
        self.N = N
        self.pred = 0

    def get_score(self,score,pred,info=None):

        if len(self.history) < self.N:
            heapq.heappush(self.history,
                           history_score(score, pred, info))
            n = len(self.history)
            self.pred = ((n - 1) * self.pred + pred) / n

            return self.pred, True
        else:
            if score > self.history[0].score:

                assert len(self.history) == self.N

                old_record = heapq.heappushpop(self.history,
                                history_score(score,pred,info))
                self.pred = self.pred + (pred - old_record.pred)/self.N

                return self.pred,True

        return pred,False
