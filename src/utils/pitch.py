from src.config import PITCH_LENGTH, PITCH_WIDTH

class Pitch:
    def __init__(self, length=PITCH_LENGTH, width=PITCH_WIDTH):
        self.length = length
        self.width = width
        self.x_min = -length / 2
        self.x_max = length / 2
        self.y_min = -width / 2
        self.y_max = width / 2

    def normalize_coordinates(self, x, y, attacking_direction='R'):
        if attacking_direction == 'L':
            x = -x
            y = -y
        
        return x, y
