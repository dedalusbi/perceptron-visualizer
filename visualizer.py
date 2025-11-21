import pygame
import sys
import settings
from perceptron import Perceptron

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((settings.WIDTH,settings.HEIGHT))
        pygame.display.set_caption("Interactive Perceptron Visualizer")
        self.clock = pygame.time.Clock()
        self.points = []
        self.perceptron = Perceptron()
    
    def handle_input(self):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if event.button == 1:
                    self.points.append({
                        'x': x,
                        'y': y,
                        'label': 0,
                        'color': settings.CLASS_0_COLOR
                    })
                elif event.button == 3:
                    self.points.append({
                        'x': x,
                        'y': y,
                        'label': 1,
                        'color': settings.CLASS_1_COLOR
                    })

                for _ in range(100):
                    self.train_model()
                
                print(f"Training concluded. Updated weights: {self.perceptron.weights}")
                

    def draw(self):
        self.screen.fill(settings.BG_COLOR)

        for point in self.points:
            pygame.draw.circle(
                self.screen,
                point['color'],
                (point['x'], point['y']),
                settings.POINT_RADIUS
            )
        
        pygame.display.flip()

    def run(self):
        while True:
            self.handle_input()
            self.draw()
            self.clock.tick(60)
    
    def pixel_to_cartesian(self, x, y):
        cart_x = (x-settings.WIDTH/2) / (settings.WIDTH/2)
        cart_y = (y-settings.HEIGHT/2) / (settings.HEIGHT/2)
        return cart_x, cart_y
    
    def train_model(self):
        for point in self.points:

            pixel_x = point['x']
            pixel_y = point['y']

            target = point['label']

            input_x, input_y = self.pixel_to_cartesian(pixel_x, pixel_y)

            self.perceptron.train(input_x, input_y, target)