# Moving a pygame window with Tkinter.
# Used code from:
#    https://stackoverflow.com/questions/8584272/using-pygame-features-in-tkinter
#    https://stackoverflow.com/questions/31797063/how-to-move-the-entire-window-to-a-place-on-the-screen-tkinter-python3

import os
import random
import tkinter as tk

window_width, window_height = 400, 500

# Tkinter Stuffs
root = tk.Tk()
embed = tk.Frame(root, width=window_width, height=window_height)
embed.pack()

os.environ['SDL_WINDOWID'] = str(embed.winfo_id())
os.environ['SDL_VIDEODRIVER'] = 'windib'  # needed on windows.

root.update()

# Pygame Stuffs
import pygame
pygame.display.init()
screen = pygame.display.set_mode((window_width, window_height))

# Gets the size of your screen as the first element in all modes.
screen_full_size = pygame.display.list_modes()[0]

# Basic Pygame loop
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True

            if event.key == pygame.K_SPACE:
                # Press space to move the window to a random location.
                r_w = random.randint(0, screen_full_size[0] / 5)
                r_h = random.randint(0, screen_full_size[1] / 5)
                root.geometry("+"+str(r_w)+"+"+str(r_h))

    # Set to green just so we know when it is finished loading.
    screen.fill((0, 220, 0))

    pygame.display.flip()

    root.update()

pygame.quit()
root.destroy()