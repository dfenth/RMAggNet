"""
A pretty cool progress bar
"""

def progress_bar_with_eta(epoch_id, batch_id, num_batches, max_pips, batch_times, val_acc=None, epoch_time=None, epoch_loss=None):
    """
    Creates a progress bar string with optional validation information.

    Parameters:
    - epoch_id (int): The current epoch number.
    - batch_id (int): The current batch number.
    - num_batches (int): The total number of batches in the epoch.
    - max_pips (int): The maximum number of progress indicator points in the progress bar.
    - batch_times (list of float): List of times taken for each batch (for predicted time remaining).
    - val_acc (float, optional): Validation accuracy for the epoch (default is None).
    - epoch_time (float, optional): Time taken for the entire epoch (default is None).
    - epoch_loss (float, optional): Loss for the entire epoch (default is None).

    Returns:
    - progress_string (str): A string representing the progress bar with optional epoch information.
    """
    progress_string = "Epoch {}: [".format(epoch_id)
    current_pip = int((max_pips/num_batches)*batch_id)+1
    progress_string += "="*current_pip + " "*(max_pips-current_pip)
    if val_acc == None:
        progress_string += "] ({}/{} - {:.4f}s left)".format(batch_id, num_batches, np.mean(batch_times)*(num_batches-batch_id))
    else:
        progress_string += "] (Val Acc: {:.4f} - took {:.4f}s - Epoch loss: {:.4f})".format(val_acc, epoch_time, epoch_loss)

    return progress_string