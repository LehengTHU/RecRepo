# Repo for Recommendation

## Quick Start

### General recommendation

Before using the general recommendation, run the following command to install the evaluator:
```bash
pushd models/General/base
python setup.py build_ext --inplace
popd
```


## File Structure
```
RecRepo
├── assets/ # images, pretrained models and other assets
    ├── recint/
        ├── pretrained
        ├── prompt
    ├── pictures/
├── data/ # datasets
    ├── General/
    ├── LLM/
    ├── Seq/
├── models/ # definition of models and RecSys
    ├── General/
        ├── base/
            ├── abstract_model.py
            ├── abstract_RS.py
            ├── utils.py
        ├── MF.py
        ├── LightGCN.py
    ├── LLM/
    ├── Seq/
├── weights/ # saved model weights during training
    ├── General/
        |── ml-1m/
            |── MF/
            |── LightGCN/
    ├── LLM/
    ├── Seq/
├── logs/ # saved logs during training
    |── test.log
    |── test2.log
├── main.py
├── README.md
├── requirements.txt
├── parse.py
├── utils.py

```

## Abstract RS

Containing the whole process of recommendation system, including data processing, model definition, training and evaluation.

Neccessary settings are intialized in the `__init__` function:
```python
class AbstractRS(nn.Module):
    def __init__(self, args) -> None:
        super(AbstractRS, self).__init__()

        self.args = args
        self.parse_args(args) # parse the args
        self.preperation() # neccessary configuration (e.g., file directory)
        self.load_data() # load the data
        self.add_additional_args() # add additional args (e.g., data information)
        self.load_model() # load the model
        self.get_loss_function() # get the loss function (optional)
        self.set_optimizer() # set the optimizer
```

The whole process is triggered by the `excute` function, including checkpoint loading, training and evaluation:
```python
    def execute(self):
        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        # restore the checkpoint
        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.device) 

        start_time = time.time()
        # train the model if not test only
        if not self.test_only:
            print("start training")
            self.train()
            # test the model
            print("start testing")
            self.model = self.restore_certain_checkpoint(self.data.best_valid_epoch, self.model, self.base_path, self.device)
        end_time = time.time()

        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d, total training cost is %.1f" % (max(self.data.best_valid_epoch, self.start_epoch), end_time - start_time)
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        print('VAL PHRASE:')
        self.evaluate(self.model, self.data.valid_data, self.device, name='valid')
        print('TEST PHRASE:')
        self.evaluate(self.model, self.data.test_data, self.device, name='test')
```

### train_one_epoch
Must be implemented in the subclass, including the training process of one epoch. This function is not implemented in the abstract class (LLM and General):
```python
    def train_one_epoch(self):
        raise NotImplementedError
```
The return value of this function must be a list of losses, which will be further documented. A typical implementation is as follows:
```python
    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, pos_weights  = batch[0], batch[1], batch[2], batch[3], batch[4]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[5]
                neg_items_pop = batch[6]

            self.model.train()
            mf_loss, reg_loss = self.model(users, pos_items, neg_items)
            loss = mf_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches, running_mf_loss/num_batches, running_reg_loss/num_batches]


```

## Abstract Model
```python

```