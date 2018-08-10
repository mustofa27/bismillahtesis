idsame = data(:,4) == data(:,5);
datasame = data(idsame,:);
tp = datasame(datasame(:,4) == 1,:);
tn = datasame(datasame(:,4) == 0,:);
datadiff = data(~idsame,:);
fn = datadiff(datadiff(:,4) == 1,:);
fp = datadiff(datadiff(:,4) == 0,:);
scatter3(tp(:,1),tp(:,2),tp(:,3),'r')
hold on;
scatter3(tn(:,1),tn(:,2),tn(:,3),'g')
hold on;
scatter3(fp(:,1),fp(:,2),fp(:,3),'y')
hold on;
scatter3(fn(:,1),fn(:,2),fn(:,3),'k')
hold on;
scatter3(maxmin(:,1),maxmin(:,2),maxmin(:,3),'b')