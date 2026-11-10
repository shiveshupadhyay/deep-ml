from typing import Tuple

def early_stopping(
    val_losses: list[float],
    patience: int, 
    min_delta: float) -> Tuple[int, int]:
    
    n_iter = len(val_losses)
    if not val_losses:
        return (-1, -1)
   
    best_epoch = 0
    best_val = val_losses[0]
    no_improvement_counter = 0


    for i, val in enumerate(val_losses):
        if i == 0: 
            continue
        if val < round((best_val - min_delta), 4):
            best_epoch = i
            best_val = val
            no_improvement_counter = 0
        else:
            no_improvement_counter +=1
        if no_improvement_counter >= patience:
            return i, best_epoch
    return len(val_losses) - 1, best_epoch