close all
clear
clc
subj = "9";
events = load("events_" + subj);
events = events(:,1);
diff = events(2:end)-events(1:end-1);
i = find(diff>20000);
plot(events);
hold on;
plot(i,events(i),'o','MarkerSize',10);
del = [0 events(1) ; events(i),(diff(i)) ; events(end) -1];
csvwrite("bad_seg_"+subj,del)%(2:end,:));