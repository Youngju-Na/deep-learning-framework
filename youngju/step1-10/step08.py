import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    # 변수의 창조자를 저장하는 함수를 추가
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]   # 해당 변수의 창조자를 찾아서 funcs 리스트에 넣습니다.

        while funcs:  # self.creator가 not None 인 경우에 반복문을 돌게됩니다.
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)   # 하나 앞의 함수를 리스트에 추가한다.


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
