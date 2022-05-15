# Connect 4 has 7 columns and 6 rows
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Connect4(object):
    GROUPS_OF_4 = 308
    COLS = 7
    ROWS = 6
    COLOR_1 = 1
    COLOR_2 = -1
    EMPTY = 0

    def __init__(self):
        self.arr = np.zeros((Connect4.ROWS, Connect4.COLS), dtype=np.int8)
        self.turn = Connect4.COLOR_1
        self.winner = None

    def __copy__(self):
        copy = Connect4()
        copy.arr = self.arr.copy()
        copy.turn = self.turn
        copy.winner = self.winner
        return copy

    def reset(self):
        self.arr = np.zeros((Connect4.ROWS, Connect4.COLS), dtype=np.int8)
        self.turn = Connect4.COLOR_1
        self.winner = None

    @property
    def observation(self):
        return self.arr

    @property
    def gameOver(self):
        return self.winner is not None

    @property
    def loser(self):
        return self.turn if self.gameOver else None

    def __repr__(self):
        return self.arr.__repr__()

    def __getitem__(self, item):
        return self.arr[item]

    def __setitem__(self, key, value):
        self.arr[key] = value

    def __colOpen(self, col):
        return Connect4.EMPTY in self[:, col]

    def __insertColor(self, color, col):
        if self.__colOpen(col):
            row = min(range(Connect4.ROWS), key=lambda r: float('inf') if self[r, col] != Connect4.EMPTY else r)
            self[row, col] = color
            return row
        raise Exception(f"Invalid insertion: Column {col} is full.")

    @property
    def openCols(self):
        return [col for col in range(Connect4.COLS) if self.__colOpen(col)]

    @property
    def legalMoves(self):
        return self.openCols

    def takeTurn(self, col):
        if self.gameOver:
            raise Exception("Game already over")
        row = self.__insertColor(self.turn, col)

        # Check for win
        rightRoom = Connect4.COLS - col - 1
        colIndices = np.array(range(max(0, col - 3), col + min(rightRoom, 3) + 1), dtype=self.arr.dtype)
        upRoom = Connect4.ROWS - row - 1
        rowIndices = np.array(range(max(0, row - 3), row + min(upRoom, 3) + 1), dtype=self.arr.dtype)

        # Check for win on row
        checkCols = self[row, colIndices]
        colView = sliding_window_view(checkCols, 4)
        if self.turn * 4 in colView.sum(axis=-1):
            self.winner = self.turn

        # Check for win on column
        if not self.gameOver:
            checkRows = self[rowIndices, col]
            rowView = sliding_window_view(checkRows, 4)
            if self.turn * 4 in rowView.sum(axis=-1):
                self.winner = self.turn
        # For context, (0, 0) represents the bottom left
        # Check for diag from bottom left to upper right win
        if not self.gameOver:
            dl = np.flip(np.diag(np.flip(self.arr[:row, :col])))
            ur = np.diag(self.arr[row + 1:, col + 1:])
            diag1 = np.concatenate([dl, [self.turn], ur])
            if len(diag1) >= 4 and self.turn * 4 in sliding_window_view(diag1, 4).sum(axis=-1):
                self.winner = self.turn
        # Check for diag from bottom right to upper left win
        if not self.gameOver:
            ul = np.flip(np.diag(np.fliplr(self.arr[row + 1:, :col])))
            dr = np.diag(np.flipud(self.arr[:row, col + 1:]))
            diag2 = np.concatenate([ul, [self.turn], dr])
            if len(diag2) >= 4 and self.turn * 4 in sliding_window_view(diag2, 4).sum(axis=-1):
                self.winner = self.turn

        self.turn = Connect4.COLOR_1 if self.turn == Connect4.COLOR_2 else Connect4.COLOR_2



def testTurns(p1Turns, p2Turns):
    game = Connect4()
    indecies = {game.COLOR_1: 0, game.COLOR_2: 0}
    turns = {game.COLOR_1: p1Turns, game.COLOR_2: p2Turns}
    while not game.gameOver:
        turn = game.turn
        game.takeTurn(turns[turn][indecies[turn]])
        indecies[turn] += 1
    winner = game.winner
    breakpoint()

def fourGroupings():
    """
    :return: The indices of all possible "4 in a row"'s grouped together.
    [0, 1, 2, 3, 4, 5, 6] is mapped to [
         [[0, 1], [2, 3]], [[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]
         ]
    For a standard Connect 4 board with 6 rows and 7 columns, there are 308 groupings
    """
    rangeArr = np.array(list(range(42)), dtype=np.int).reshape((6, 7))

    groupings = []
    append = lambda x: groupings.append(np.array([x.T]))
    for row in range(6):
        for col in range(7):
            rightRoom = Connect4.COLS - col - 1
            colIndices = np.array(range(max(0, col - 3), col + min(rightRoom, 3) + 1), dtype=np.int)
            upRoom = Connect4.ROWS - row - 1
            rowIndices = np.array(range(max(0, row - 3), row + min(upRoom, 3) + 1), dtype=np.int)

            # Check for win on row
            checkCols = rangeArr[row, colIndices]
            colView = sliding_window_view(checkCols, 4)

            append(colView)

            # Check for win on column

            checkRows = rangeArr[rowIndices, col]
            rowView = sliding_window_view(checkRows, 4)

            append(rowView)


            dl = np.flip(np.diag(np.flip(rangeArr[:row, :col])))
            ur = np.diag(rangeArr[row + 1:, col + 1:])
            diag1 = np.concatenate([dl, [rangeArr[row, col]], ur])
            if len(diag1) >= 4:
                diag1View = sliding_window_view(diag1, 4)
                append(diag1View)
                # groupings = np.concatenate([groupings, diag1View])


            # Check for diag from bottom right to upper left win
            ul = np.flip(np.diag(np.fliplr(rangeArr[row + 1:, :col])))
            dr = np.diag(np.flipud(rangeArr[:row, col + 1:]))
            diag2 = np.concatenate([ul, [rangeArr[row, col]], dr])
            if len(diag2) >= 4:
                diag2View = sliding_window_view(diag2, 4)
                # groupings = np.concatenate([groupings, diag2View])
                append(diag2View)
    groupings = np.concatenate(groupings, axis=-1)
    return groupings




if __name__ == '__main__':
    fourGroupings()
    # Test game
    game = Connect4()
    # Horizontal win
    p1ColWinTurns = [1, 2, 3, 4]
    p2ColWinTurns = [0, 1, 0]
    # testTurns(p1ColWinTurns, p2ColWinTurns)
    # Vertical Win
    p1RowWinTurns = [1, 2, 3, 3]
    p2RowWinTurns = [1, 1, 1, 1]
    # testTurns(p1RowWinTurns, p2RowWinTurns)

    # p1DiagTest = [3, 2, 2]
    # p2DiagTest = [3, 2, 2]
    # p1DiagTest = [3, 3, 3]
    # p2DiagTest = [3, 3, 3]
    # testTurns(p1DiagTest, p2DiagTest)
    # UR Diag Win
    p1URDiagWinTurns = [3, 4, 5, 5, 6, 6]
    p2URDiagWinTurns = [4, 5, 6, 6, 0]
    testTurns(p1URDiagWinTurns, p2URDiagWinTurns)
