from torch import no_grad


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count
