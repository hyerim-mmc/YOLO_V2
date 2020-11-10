 def lr_scheduling(self, step):
        if step < 1001 and self.burn_in:
            lr = self.lr * (step/1004)**4
            for g in self.optimizer.param_groups:
                g['lr'] = lr
    
    def run(self):
        step = 1
        print_interval = 100
        for i in range(self.epoch):
            Loss = []
            Prec = []
            Val_Loss = []
            t_Prec = []
            n = 0
            for data in self.dataset:
                self.network.train()
                image, label = data[0].to(self.device), data[1].to(self.device)
                
                hypo = self.network.forward(image)
                loss = self.critieron(hypo, label)
                loss.backward()
                n += 1
                if n == self.division:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduling(step)
                    n = 0
                    step += 1

                with torch.no_grad():
                    Loss.append(loss.detach().cpu().numpy())
                    total_num = len(label)
                    idx = torch.argmax(hypo, dim=1)
                    total_true = (idx == label).float().sum()
                    t_precision = total_true/total_num
                    t_Prec.append(t_precision.cpu().detach().numpy())
                
                if step > 1000:
                    print_interval = 100
                
                if step % print_interval == 0 and n == 0:
                    with torch.no_grad():
                        self.network.eval()
                        k = 0
                        for val_data in self.val_dataset:
                            val_image, val_label =\
                                val_data[0].to(self.device), val_data[1].to(self.device)
                            val_hypo = self.network.forward(val_image)
                            val_loss = self.critieron(val_hypo, val_label)
                            idx = torch.argmax(val_hypo, dim=1)
                            total_num = len(val_label)
                            total_true = (idx == val_label).float().sum()
                            precision = total_true/total_num
                            Prec.append(precision.cpu().numpy())
                            Val_Loss.append(val_loss.cpu().numpy())
                            k += 1 
                            if k == 10:
                                break
                            
                    loss = np.array(Loss).mean()
                    val_loss = np.array(Val_Loss).mean()
                    prec = np.array(Prec).mean()
                    tprec = np.array(t_Prec).mean()
                    print("""Epoch: {} // Step: {} // Loss : {:.2f} // Val_Loss : {:.2f} //
                           Prec : {:.3f} // Val_Prec : {:.3f}""".format(
                               (i+1), step, loss, val_loss, tprec, prec))
                    Loss = []
                    Prec = []
                    Val_Loss = []
                    t_Prec = []
            save_path = './dataset/Darknet19.pth'
            torch.save(self.network.state_dict(), save_path)