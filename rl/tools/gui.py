import io

import ipywidgets
import PIL, PIL.Image
import bqplot, bqplot.pyplot
import time
import numpy as np

class GameGUI():
    actionMap = {'left': 'arrow-left',
                 'right': 'arrow-right', 
                 'up': 'arrow-up', 
                 'down': 'arrow-down', 
                 'forward': 'arrow-up', 
                 'rotateleft': 'rotate-left', 
                 'rotateright': 'rotate-right', 
                 'skip': 'dot-circle-o'}
    blay = ipywidgets.Layout(width='34px', height='30px', border='1px solid black')
    ilay = ipywidgets.Layout(width='80px', height='30px', border='1px solid black')
    scoreFmt = "<center><b>{:2.2f} / {:2.2f}</b></center>"
    
    def __init__(self, game, nb_frames=1, model=None):
        self.game = game
        self.nb_frames = nb_frames
        self.model = model
        self.game.reset()
        initFrame = self.game.get_frame()
        self.state = np.repeat(np.expand_dims(initFrame, axis=0), self.nb_frames, axis=0)

        self.actions = {self.actionMap[v]: k for k,v in self.game.actions.items()}
        initScore = self.game.get_score()
        self.aggScore = initScore
        self._initGamePlot(initFrame, initScore)
        self._initQPlot()

    def _onButtonClicked(self, args):
        if args.description == 'New':
            self.game.reset()
            self.stat.value = ""
            self.aggScore = 0.0
            initFrame = self.game.get_frame()
            self.state = np.repeat(np.expand_dims(initFrame, axis=0), self.nb_frames, axis=0)
            self.stat.value = self.scoreFmt.format(0.0, self.aggScore)
            self._plotGame(initFrame)
            self._updateValues()

        elif not self.game.is_over():
            args.style.button_color = 'red'
            time.sleep(0.1)
            args.style.button_color = 'yellow'
            currentFrame, currentScore, game_over = self.game.play(self.actions[args.icon])
            self.state = np.append(self.state[1:], np.expand_dims(currentFrame, axis=0), axis=0)
            self.aggScore += currentScore
            self.stat.value = self.scoreFmt.format(currentScore, self.aggScore)
            self._plotGame(currentFrame)
            self._updateValues()

    def _initGamePlot(self, initFrame, initScore):
        bnew = ipywidgets.Button(layout=self.ilay, style=ipywidgets.ButtonStyle(font_weight='bold', button_color='green'), description = 'New')
        self.stat = ipywidgets.HTML(layout=self.ilay, value=self.scoreFmt.format(initScore, self.aggScore))
        
        controls = [bnew]
        for _, i in sorted(tuple(self.game.actions.items())):
            button = ipywidgets.Button(layout=self.blay, style=ipywidgets.ButtonStyle(font_weight='bold', button_color='yellow'), icon=self.actionMap[i])
            controls.append(button)
        for c in controls:
            c.on_click(self._onButtonClicked)
        controls.append(self.stat)
        self.ctrlbox = ipywidgets.HBox(controls)
                
        self.canvas = ipywidgets.Image()
        self.imbuf = io.BytesIO()
        self._plotGame(initFrame)
                
        ibox = ipywidgets.VBox([self.canvas, self.ctrlbox])
        ibox.layout.align_items = 'center'
        self.gamebox = ipywidgets.HBox([ibox])

    def _initQPlot(self):
        if self.model is None: return
        self.qbox = bqplot.pyplot.figure()
        self.qbox.layout = {"height": "256px", "width": "256px"}
        self.qbox.fig_margin = {"top":10, "bottom":20, "left":30, "right":0}
        q_values = list(self.model(self.state[np.newaxis], training=False).numpy()[0])
        action_names = [self.game.actions[i] for i in range(self.game.nb_actions)]
        bqplot.pyplot.bar(action_names, q_values)
        bqplot.pyplot.hline(0)
        self.qbox.axes[0].tick_style = {'font-size': 14, 'font-weight': 'bold'}
        self.qbox.axes[1].tick_style = {'font-size': 14, 'font-weight': 'bold'}

    def _plotGame(self, frame):
        self.imbuf.seek(0)
        fx, fy = frame.shape[0], frame.shape[1]
        rx, ry = (256, int(fy*256/fx)) if (fx > fy) else (int(fx*256/fy), 256)
        PIL.Image.fromarray((frame*255).astype('uint8')).resize((ry, rx), resample=PIL.Image.NEAREST).save(self.imbuf, 'gif')
        self.canvas.value = self.imbuf.getvalue()
    
    def _updateValues(self):
        if self.model is None: return
        q_values = list(self.model(self.state[np.newaxis], training=False).numpy()[0])
        self.qbox.marks[0].y = q_values
