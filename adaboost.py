import numpy as np


def calc_error(y, y_):
    return (y - y_) ** 2


class AdaBoost:
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.N = len(self.trainX)
        self.weights = np.ones(self.N) / self.N
        self.alphas = []
        self.models = []

    def set_rule(self, model):
        predict = model.predict(self.trainX)
        errors = []
        for i in range(self.N):
            errors.append(calc_error(self.trainY[i], predict[i]))
        e = (errors * self.weights).sum()
        alpha = 0.5 * np.log((1 - e) / e)
        print('e=%.4f a=%.4f' % (e, alpha))
        w = np.zeros(self.N)
        for i in range(self.N):
            w[i] = self.weights[i] * np.exp(alpha * errors[i] / e)
        self.weights = w / w.sum()
        self.models.append(model)
        self.alphas.append(alpha)

    def predict(self, x_set):
        n_models = len(self.models)
        alpha_sum = np.sum(self.alphas)
        final_predict = np.zeros(len(x_set))
        for i in range(n_models):
            predict = self.models[i].predict(x_set)
            final_predict = final_predict + predict * self.alphas[i]
        final_predict = final_predict / alpha_sum
        return final_predict

    def evaluate(self):
        n_models = len(self.models)
        alpha_sum = np.sum(self.alphas)
        final_predict = np.zeros(len(self.trainX))
        for i in range(n_models):
            predict = self.models[i].predict(self.trainX)
            final_predict = final_predict + predict * self.alphas[i]
        final_predict = final_predict / alpha_sum
        errors = []
        for i in range(self.N):
            errors.append(calc_error(self.trainY[i], final_predict[i]))
        return np.sum(errors)

    def get_weights(self):
        return self.weights
