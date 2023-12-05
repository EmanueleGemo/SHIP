%% INTRO
% This script summarizes the fit process of the experimental data shown 
% in Fig 4 of 10.1002/aisy.202200145.
% Fig1: datapoints represent the digitized data, solid lines are their linear fit (using a xlog scale), 
% and the dashed lines represent the final model result.
% Fig3: same as Fig 1
% Fig2 and Fig4 show the fitting process
% Fig 5 correlates the values of (\mu)0 and (\mu/\sigma)0
% DataPoints:
% Each datapoint digitizes the experimental data portion of the plot.
% there presented. A total of 4 points per dataset is collected.
% the digitization collects the data sequentially, hence the unconventional
% format that we need to process iteratively.
% We assume that lines fit the data (plotted in xlog scale).

%% GO
clear all %#ok<CLALL>
clc

mean_raw_data = [
58.37592378488597, 44.58077709611453
145.63484775012444, 42.78118609406954
683.72176694842, 40
3840.7459778154425, 37.218813905930475
58.37592378488597, 53.25153374233129
151.99110829529332, 51.615541922290404
672.1377007734901, 49.325153374233125
3840.7459778154425, 47.68916155419224
59.891541564587094, 60.77709611451944
155.93726299663388, 60.77709611451944
683.72176694842, 60.28629856850716
3775.6735203936405, 59.46830265848671
59.891541564587094, 71.57464212678937
159.98587196060575, 70.920245398773
683.72176694842, 70.920245398773
3775.6735203936405, 70.920245398773
59.382013107607534, 81.88139059304703
161.35863368107667, 81.39059304703477
695.5054805899201, 81.06339468302659
3775.6735203936405, 80.57259713701433
59.891541564587094, 91.86094069529653
161.35863368107667, 91.53374233128835
701.4732781737384, 91.20654396728017
3775.6735203936405, 90.87934560327199
59.382013107607534, 102.65848670756645
166.96848567866635, 102.1676891615542
695.5054805899201, 102.1676891615542
3775.6735203936405, 102.1676891615542
58.37592378488597, 116.56441717791411
168.4011618447284, 116.56441717791411
701.4732781737384, 116.07361963190183
3775.6735203936405, 116.07361963190183
];

std_raw_data = [
59.72184417164956, 0.17321178120617
249.90781275313626, 0.234642356241
944.111150069066, 0.29523141654978
2602.2927495953313, 0.33478260869565
61.26804780573786, 0.1025245441795233
99.57489594326087, 0.116830294530154
993.630220712639, 0.18751753155680
2982.363613316007, 0.2237026647966
61.26804780573786, 0.05455820476
99.57489594326087, 0.059607293127629
976.8418036096749, 0.08148667601683
3506.4355837219855, 0.0949509116409538
3506.4355837219855, 0.09495091164095387
61.26804780573786, 0.0596072931276299
98.7301015513369, 0.0654978962131839
1002.1323211350491, 0.0983169705469
3536.438764821014, 0.1151472650771390
60.74824908839928, 0.0410939691444602
99.57489594326087, 0.04782608695652196
1002.1323211350491, 0.06129032258064
3566.698671262515, 0.0713884992987379
61.26804780573786, 0.0318373071528753
99.57489594326087, 0.03183730715287536
1002.1323211350491, 0.03772791023842936
3627.9974657576668, 0.04277699859747563
60.74824908839928, 0.025946704067321358
101.28623229260847, 0.028471248246844494
1002.1323211350491, 0.02931276297335228
3659.0408037551824, 0.03099579242636763
60.74824908839928, 0.044460028050491096
98.7301015513369, 0.042776998597475635
993.630220712639, 0.04530154277699877
3627.9974657576668, 0.0461430575035065
];

%% processing the averages

figure(1)
clf(1)
hold on
set(gca,'ColorOrder',parula(ceil(size(mean_raw_data,1))/4))

idx = 1;
idx_color = 1;
mean_arranged_data = [];

t_init = 10;%seconds
t_pinch = 1;%sec # <--- lines are "pinched" so to fit this common point as well
m_ave = [];
q_ave = [];
G1s_ave = []; % Initial value, at 1 s. 
% Fits using a lower extrapolated value (1 ms) yield worse results,
% possibly because far away from the fitting area.
% The iterations retrieve the G value at 1 s for each line (linear fit of 4
% datapoints per dataset).

while idx < size(mean_raw_data,1)
    mean_arranged_data = [mean_arranged_data, mean_raw_data(idx:idx+3,:)]; %#ok<*AGROW>
    
    set(gca,'ColorOrderIndex',idx_color)
    scatter(mean_raw_data(idx:idx+3,1),mean_raw_data(idx:idx+3,2),...
        'o','filled')
    
    fr = fit(log10(mean_raw_data(idx:idx+3,1)),...
        mean_raw_data(idx:idx+3,2),'poly1');
    
    m_ave = [m_ave;fr.p1];
    q_ave = [q_ave;fr.p2];

    set(gca,'ColorOrderIndex',idx_color)
    t = logspace(0,5,10)';
    plot(t,fr.p1*log10(t)+fr.p2,'-')
    
    G1s_ave = [G1s_ave;fr.p1*log10(t_init)+fr.p2];
    
    idx = idx + 4;
    idx_color = idx_color + 1;
end
ax = gca;
ax.XScale = 'log';
ax.XLabel.String = 'Time (s)';
ax.YLabel.String = 'Conductance \mu (\muS)';

%% getting m and q for the average
figure(2)
clf(2)
ax1 = subplot(211);
hold(ax1,'on')

scatter(G1s_ave,m_ave,'o','filled','DisplayName',"exp. \\mu linear fit (slopes)")
ax1.XLabel.String = '\mu (\muS)';
ax1.YLabel.String = 'm for \mu';

% here we split the fit, in that we get better results using two linear
% fits of a lower part and an upper part of the datapoints
m = [true,true,true,false,false,false,false,false];
[fmw1,G1] = fit(G1s_ave(m),m_ave(m),'poly1');
const = fittype('p2+0*x',...
    'dependent',{'y'},'independent',{'x'},...
    'coefficients',{'p2'});
[fmw2,G2] = fit(G1s_ave(~m),m_ave(~m),const);

rr = linspace(50,120,7001);
intersect = rr(find(feval(fmw1,rr)>feval(fmw2,rr),1,'first')) %#ok<NOPTS> 
% linear fits intersect at this value; we use this as a toggle to select
% either of the linear fits (given the as-written G value).


str1 = sprintf('G0>=%0.2f\\muS -> fit: y = %0.2fx%0.2f , R² = %0.2f',intersect,fmw1.p1,fmw1.p2,G1.rsquare);
rss = sum((m_ave(~m)-fmw2.p2).^2);
tss = sum((m_ave(~m)-mean(m_ave(~m))).^2); %<- of course  R² is 1 here
str2 = sprintf('G0<%0.2f\\muS -> fit: y = %0.2f , R² = %0.2f',intersect,fmw2.p2, G2.sse/tss);
plot(rr,feval(fmw1,rr),'-','DisplayName',str1)
plot(rr,feval(fmw2,rr),'-','DisplayName',str2)
legend('location','best')

ax2 = subplot(212);
hold(ax2,'on')
scatter(G1s_ave,q_ave,'o','filled','DisplayName',"exp. \\mu linear fit (intercepts)")
ax2.XLabel.String = '\mu (\muS)';
ax2.YLabel.String = 'q  for \mu';
[fqw,G1] = fit(G1s_ave,q_ave,'poly1');
str1 = sprintf('y = %0.2fx%0.2f , R² = %0.2f',fqw.p1,fqw.p2,G1.rsquare);
plot(rr,feval(fqw,rr),'-','DisplayName',str1)
legend('location','best')

%% std processing
figure(3)
clf(3)
hold on
set(gca,'ColorOrder',parula(ceil(size(std_raw_data,1))/4))
idx = 1;
idx_color = 1;

G1s_std = [];
while idx < size(std_raw_data,1)
    
    set(gca,'ColorOrderIndex',idx_color)
    scatter(std_raw_data(idx:idx+3,1),std_raw_data(idx:idx+3,2),...
        'o','filled')
    
    fr = fit(log10(std_raw_data(idx:idx+3,1)),...
        std_raw_data(idx:idx+3,2),'poly1');
    
    set(gca,'ColorOrderIndex',idx_color)
    t = logspace(0,5,10)';
    plot(t,fr.p1*log10(t)+fr.p2,':','linewidth',0.5)
    
    G1s_std = [G1s_std;fr.p1*log10(t_pinch)+fr.p2];
    
    idx = idx + 4;
    idx_color = idx_color + 1;
end
mstd = mean(G1s_std);

ax = gca;
ax.XScale = 'log';
ax.XLabel.String = 'Time (s)';
ax.YLabel.String = 'std \sigma over conductance \mu, \sigma/\mu';

%% pinching fits at t = 1 s
m_std = [];
q_std = [];
t = logspace(0,5,10)';
idx = 1;
idx_color = 1;
G1s_std = [];
fo = fitoptions('weights',[5,1,1,1,1]);
while idx < size(std_raw_data,1)
    
    fr2 = fit(log10([t_pinch;std_raw_data(idx:idx+3,1)]),...
        [mstd;std_raw_data(idx:idx+3,2)],'poly1',fo);
    
    m_std = [m_std;fr2.p1];
    q_std = [q_std;fr2.p2];
    
    set(gca,'ColorOrderIndex',idx_color)    
    plot(t,feval(fr2,log10(t)),'-','linewidth',1)
    
    G1s_std = [G1s_std;feval(fr2,log10(t_init))];
    
    idx = idx + 4;
    idx_color = idx_color + 1;
end

%% calculating params for std
figure(4)
clf(4)
ax1 = subplot(211);
hold(ax1,'on')
scatter(G1s_std,m_std,'o','filled','DisplayName',"exp. \\sigma linear fit (slopes)")
% ax1.YScale = 'log';
ax1.XLabel.String = '\sigma/\mu';
ax1.YLabel.String = 'm for \sigma/\mu';

[a,b] = sortrows(G1s_std);
[~,outlier] = max(m_std(b));
m = true(length(G1s_std),1); m(outlier) = false;
b = m_std(b);
[fms,G1] = fit(a(m),b(m),'poly1');
rr = linspace(min(G1s_std),max(G1s_std),numel(G1s_std)*10);
str1 = sprintf('y = %0.2fx%0.2f , R² = %0.2f',fms.p1,fms.p2,G1.rsquare);
plot(rr,feval(fms,rr),'-','DisplayName',str1)
legend('location','best')


ax2 = subplot(212);
hold(ax2,'on')
scatter(G1s_std,q_std,'o','filled','DisplayName','exp. \\sigma linear fit (intercepts)')
% ax2.YScale = 'log';
ax2.XLabel.String = '\sigma/\mu';
ax2.YLabel.String = 'q  for \sigma/\mu';

[fqs,G2] = fit(G1s_std,q_std,'poly1');
str1 = sprintf('y = %0.2fx+%0.2f , R² = %0.2f',fqs.p1,fqs.p2,G2.rsquare);
plot(G1s_std,feval(fqs,G1s_std),'-','DisplayName',str1)
legend('location','best')


figure(5)
clf(5)
ax1 = subplot(1,1,1);
hold on;
% % linear, t_init = 1 sec
% fo = fitoptions('weights',[1,10,1,1,1,1,1,10]);
% fG = fit(G1s_ave,G1s_std,'poly1',fo);
% hyperbole, t_init = 10 sec
hyper = fittype('a+b/(x-c)',...
    'dependent',{'y'},'independent',{'x'},...
    'coefficients',{'a','b','c'});
[fG,G1] = fit(G1s_ave,G1s_std,hyper);
scatter(ax1,G1s_ave,G1s_std,'DisplayName','\mu vs \sigma/\mu (calculated)')
str1 = sprintf('y = %0.2f+%0.2f/(x-%0.2f) , R² = %0.2f',fG.a,fG.b,fG.c,G2.rsquare);
plot(ax1,G1s_ave,feval(fG,G1s_ave),'-','DisplayName',str1)
legend('location','best')
ax1.XLabel.String = '\mu (\muS)';
ax1.YLabel.String = '\sigma/\mu';

times = [60,4000];
w0 = G1s_ave;%[117.548,102.335,92.4735,81.7176,70.7071,62.7156,57.8699,52.1887];
newmu = zeros(length(w0),length(times));
news_mu = zeros(length(w0),length(times));
ii = 0;
for t = times
    ii = ii+1;
    jj = 0;
    for ww = w0'
        jj = jj+1;
        if t>65.099
            newmu(jj,ii) = (0.2882*ww-19.17)*t+ww;
        else
            newmu(jj,ii) = -0.4129*t;
        end
        news_mu(jj,ii) = (4.651*(0.00131*ww-0.09859).^2 ...
           -.7823*(0.00131*ww-0.09859)+0.032606)*t...
           +(0.00131*ww-0.09859);
        % newmu(jj,ii) = fmu(ww,t);
        % news_mu(jj,ii) = fs_mu(ww,t);
    end
end
% 
figure(1)
hold on
ii = 0;
times = [10,10000];
for ww = w0'
    ii = ii+1;
    plot(times,mut(ww,log10(times)),'--')
end
figure(3)
hold on
ii = 0;
times = [10,10000];
for ww = w0'
    ii = ii+1;
    plot(times,smt(sm0(ww),log10(times)),'--')
end
print('done')

function out = mut(w0,log10t)
    if w0<64.409999999999997
        out = (0.2302*w0-15.24)*log10t+0.9546*w0+4.82;
    else
        out = -0.4129*log10t+w0;
    end
end

function out = smt(sm0,log10t)
    out = (1.047*sm0-0.006294)*log10t-0.01677*sm0+0.005473;
end

function out = sm0(w0)
    out = 0.005127+0.85543./(w0-38.72);
end
function out = wt(w0,t)
    log10t = log10(t);
    sm0 = 0.005127+0.85543./(w0-38.72);
    sel = w0<64.41;
    mu = sel*((0.2302*w0-15.24)*log10t+0.9546*w0+4.82)+(1-sel)*(-0.4129*log10t+w0);
    smu = (1.047*sm0-0.006294)*log10t-0.01677*sm0+0.005473;
    out = normrnd(mu,mu*smu);    
end
