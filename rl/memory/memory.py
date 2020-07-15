#!/usr/bin/env python3

class Memory:
    """
    Abstract base class to built new replay buffers
    """
    def __init__(self): pass
    
    def remember(self, S, a, r, S_next, game_over): 
        """
        Store one record in the replay buffer.
        
        # Arguments
            S:          Game state directly before the action. Numpy array in format `(nb_frames, height, width, channels)`
            a:          Integer. Played action.
            r:          Float. Received reward.
            S_next:     Game state after the action. Same format as `S`.
            game_over:  Boolean. Game over in this state.
        """
        pass

    def get_batch(self, batch_size): 
        """
        Get a batch of replay records.
        
        # Arguments
            batch_size:  Integer. Batch size.

        # Returns
            Batch of replay records. Format of one records see `remember`
        """
        pass

    def update(self, batch):
        """
        Update memory structure after training
        
        # Arguments
            batch: Training batch
        
        # Return
            none
        """
        pass

    def reset(self): 
        """Flush and empty the replay buffer"""
        pass

if __name__ == '__main__':
    pass

