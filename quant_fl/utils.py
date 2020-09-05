

class AutoStep():
    def __init__(self, func, name):
        self.func = func
        self.step = 0
        self.name = name
    def write(self, val):
        self.func(self.name, val, self.step)
        self.step += 1   