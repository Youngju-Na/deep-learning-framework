import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    # 변수의 창조자를 저장하는 함수를 추가
    def set_creator(self, func):
        self.creator = func

    # 자동 미분
    def backward(self):
        f = self.creator   # Variable의 창조자를 알아냅니다.
        if f is not None:  # 만약 창조자가 존재하면
            x = f.input    # 함수의 input을 알아내고
            x.grad = f.backward(self.grad) # 그 input의 grad를 계산합니다.
            x.backward()


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 해당 함수에 의해 만들어진 결과임을 표시
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy

        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

if __name__ == "__main__":

    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # backward
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
