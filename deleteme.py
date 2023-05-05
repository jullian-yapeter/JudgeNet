class Game():
    def __init__(self, x,y):
        self.board = [[0]*x for _ in range(y)]
        print(self.board)

game = Game(3,4)