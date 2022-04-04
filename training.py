import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import copy
import time
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

from data import ArithmeticDataset
from model import TransformerModel, generate_square_subsequent_mask
from helpers import visualize_attention, make_plot

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = args.batch_size
    emsize = args.embed  # embedding dimension
    d_hid = args.width  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.layers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = args.heads  # number of heads in nn.MultiheadAttention
    dropout = args.dropout  # dropout probability
    print(f'Dropoput: {dropout}')
    lr = args.lr
    print(f'LR: {lr}')
    log_interval = args.log_interval
    betas = (args.beta1, args.beta2)
    weight_decay = args.weight_decay
    print(f'Weight decay: {weight_decay}')
    op = args.op
    print("Operation: ", op)
    mod = args.mod

    dset = ArithmeticDataset(op, mod)
    print('Dataset len: ', len(dset))
    ntokens = len(dset.vocab.stoi)
    train_len = int(0.5 * len(dset))
    train_dset, test_dset = random_split(dset, [train_len, (len(dset)-train_len)])
    print(f'Train len: {len(train_dset)}, test len: {len(test_dset)}')
    print("Batch size: ", batch_size)
    train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=True)
    print(f'Train batches: {len(train_dataloader)}, test batches: {len(test_dataloader)}')

    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, norm_first=False).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    print(f'Optimizer: {optimizer}')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    src_mask = generate_square_subsequent_mask(5).to(device)

    def train(model: nn.Module) -> None:
        model.train()  # turn on train mode
        total_loss = 0.
        total_accuracy = 0.

        for batch, data in enumerate(train_dataloader, 0):
            data = data.transpose(0,1)
            targets = data[-1].to(device)
            inputs = data[:-1].to(device)

            #add gaussian noise
            # with torch.no_grad():
            #     for name, param in model.named_parameters():
            #         if 'weight' in name:
            #             param.add_((torch.randn(param.size()) * 0.01).to(device))

            output = model(inputs, src_mask)
            pred = output[-1, ...].to(device)
            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            total_accuracy += (torch.argmax(output, dim=-1)[-1] == targets).sum() / len(targets)

        return total_loss / len(train_dataloader), total_accuracy.item() / len(train_dataloader) 

    def evaluate(model: nn.Module) -> float:
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        total_accuracy = 0.
        with torch.no_grad():
            for batch, data in enumerate(test_dataloader, 0):
                data = data.transpose(0,1)
                targets = data[-1].to(device)
                inputs = data[:-1].to(device)
                output = model(inputs, src_mask)
                pred = output[-1, ...].to(device)
                total_loss += batch_size * criterion(pred, targets).item()
                total_accuracy += (torch.argmax(output, dim=-1)[-1] == targets).sum()
        return total_loss / len(test_dataloader.dataset), total_accuracy.item() / len(test_dataloader.dataset)

    best_val_loss = float('inf')
    epochs = int(10e4)
    best_model = None
    train_reached_99, test_reached_99 = False, False
    loss, acc = {'train': [], 'val': []}, {'train': [], 'val': []}
    mean_weights = {'embed': [], 'attn0': [], 'attn1': [], 'decoder': []}
    # writer = SummaryWriter()
    # sample_batch = next(iter(train_dataloader)).transpose(0,1)[:-1].to(device)
    # writer.add_graph(model, (sample_batch, src_mask))

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train(model)
        val_loss, val_acc = evaluate(model)
        elapsed = time.time() - epoch_start_time
        if epoch % log_interval == 0:
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                    f'train loss {train_loss:5.2f} | train accuracy {train_acc:5.2f} |'
                    f'valid loss {val_loss:5.2f} | valid accuracy {val_acc:5.2f}')
            print('-' * 89)

        loss['train'].append(train_loss)
        loss['val'].append(val_loss)
        acc['train'].append(train_acc)
        acc['val'].append(val_acc)

        # writer.add_scalars('Epoch Loss', {'train': train_loss, 'test': val_loss}, epoch)
        # writer.add_scalars('Epoch Accuracy', {'train': train_acc, 'test': val_acc}, epoch)

        # writer.add_histogram("Embedding weights", model.encoder.weight, epoch)

        # for layer_id, layer in enumerate(model.transformer_encoder.layers):
        #     writer.add_histogram(f'Attention layer {layer_id + 1}', layer.self_attn.out_proj.weight, epoch)

        # writer.add_histogram("Decoder bias", model.decoder.bias, epoch)
        # writer.add_histogram("Decoder weights", model.decoder.weight, epoch)

        mean_weights['embed'].append(model.encoder.weight.mean().item())
        for layer_id, layer in enumerate(model.transformer_encoder.layers):
            mean_weights[f'attn{layer_id}'].append(layer.self_attn.out_proj.weight.mean().item())
        mean_weights['decoder'].append(model.decoder.weight.mean().item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        if train_acc >= 1.0 and not train_reached_99:
            torch.save(model.state_dict(), f'./models/model_before_grokking_epoch_{epoch}_steps_{epoch * len(train_dataloader)}.pt')
            train_reached_99 = True
            print('\nReached 100% on training data', f'\nepoch {epoch}, steps {epoch * len(train_dataloader)}')

        if val_acc >= 1.0 and not test_reached_99:
            torch.save(model.state_dict(), f'./models/model_after_grokking_epoch_{epoch}_steps_{epoch * len(train_dataloader)}.pt')
            test_reached_99 = True
            print('\nReached 100% on test data', f'\nepoch {epoch}, steps {epoch * len(train_dataloader)}')

        if epoch % (log_interval*10) == 0:
            steps = torch.arange(len(acc['train'])).numpy() * len(train_dataloader)

            make_plot([(steps, acc['train'], "train"), (steps, acc['val'], "val")], 
                "lr warmup first 10 epochs", "Optimization Steps", 
                "Accuracy", f'./figures/stats/acc/acc_epoch_{epoch}.png')
            
            make_plot([(steps, loss['train'], "train"), (steps, loss['val'], "val")], 
                "lr warmup first 10 epochs", "Optimization Steps", 
                "Loss", f'./figures/stats/loss/loss_epoch_{epoch}.png')
            
            make_plot([(steps, mean_weights['embed'], 'embedding'),
                     (steps, mean_weights['attn0'], 'attention 1'),
                     (steps, mean_weights['attn1'], 'attention 2'),
                     (steps, mean_weights['decoder'], 'decoder')], 
                "lr warmup first 10 epochs", "Optimization Steps", 
                "Magnitude", f'./figures/stats/weights/weights_epoch_{epoch}.png')

            sample = next(iter(train_dataloader))
            source = dset.vocab.decode(sample[0]).split()[:-1]
            target = dset.vocab.decode(sample[0]).split()[1:]
            visualize_attention(model, source, target, epoch)

            np.save( f'./stats/loss_epoch_{epoch}.npy', loss)
            np.save( f'./stats/acc_epoch_{epoch}.npy', acc)
            np.save( f'./stats/weights_epoch_{epoch}.npy', mean_weights)
            # to load: np.load(path, allow_pickle=True).item()

    steps = torch.arange(len(acc['train'])).numpy() * len(train_dataloader)
    make_plot([(steps, acc['train'], "train"), (steps, acc['val'], "val")], 
                "lr warmup first 10 epochs", "Optimization Steps", 
                "Accuracy", f'./figures/stats/acc/acc_epoch_{epoch}.png')
            
    make_plot([(steps, loss['train'], "train"), (steps, loss['val'], "val")], 
        "lr warmup first 10 epochs", "Optimization Steps", 
        "Loss", f'./figures/stats/loss/loss_epoch_{epoch}.png')
    
    make_plot([(steps, mean_weights['embed'], 'embedding'),
                (steps, mean_weights['attn0'], 'attention 1'),
                (steps, mean_weights['attn1'], 'attention 2'),
                (steps, mean_weights['decoder'], 'decoder')], 
        "lr warmup first 10 epochs", "Optimization Steps", 
        "Magnitude", f'./figures/stats/weights/weights_epoch_{epoch}.png')

    print('\n\nSaving stats...')
    np.save( f'./stats/loss_train.npy', loss['train'])
    np.save( f'./stats/loss_val.npy', loss['val'])
    np.save( f'./stats/acc_train.npy', acc['train'])
    np.save( f'./stats/acc_val.npy', acc['val'])
    np.save( f'./stats/weights_embed.npy', mean_weights['embed'])
    np.save( f'./stats/weights_attn0.npy', mean_weights['attn0'])
    np.save( f'./stats/weights_attn1.npy', mean_weights['attn1'])
    np.save( f'./stats/weights_decoder.npy', mean_weights['decoder'])

    # writer.close()

    # save best model
    best_model.eval()
    torch.save(best_model.state_dict(), './models/model.pt')

    # save trained model
    model.eval()
    torch.save(model.state_dict(), './models/model_after_training.pt')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mod", type=int, default=97)
    parser.add_argument("--op", type=str, default='/')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()
    main(args)