clear;
clc;
addpath('tool/', 'dataset/', 'result/', 'user/');

load BBCSport.mat
X{1} = data{1}; %m*n
X{2} = data{2}; 
truth = double(truelabel{1}'); % n*1

% 数据集相关信息
maxiter = 1000;
class_num = length(unique(truth)); % 类别个数

% 调整参数
alpha = 10;
beta = 1000;

fileID = fopen('result/BBCSport_DiNMF_tune.txt','a');
replic = 10;

% 指标
AC_ = zeros(1, replic);
NMI_ = zeros(1, replic);
purity_ = zeros(1, replic);
Fscore_ = zeros(1, replic);
Precision_ = zeros(1, replic);
Recall_ = zeros(1, replic);
AR_ = zeros(1, replic);
% 记录矩阵
res_record = zeros(7, replic); % 1.clustering result 2.objective function value 3.V_star
V_star_record = cell(1, replic);

for i = 1: replic
    % 运行DiNMF
    [V_star, obj] = DiNMF_update(X,truth, 'alpha', alpha, 'beta', beta, 'maxiter', maxiter);
    idx = litekmeans(V_star', class_num, 'Replicates', 20); % 执行1次kmeans
    result = EvaluationMetrics(truth, idx);
    AC_(i) = result(1);
    NMI_(i) = result(2);
    purity_(i) = result(3);
    Fscore_(i) = result(4);
    Precision_(i) = result(5);
    Recall_(i) = result(6);
    AR_(i) = result(7);
    res_record(:, i) = result;
    V_star_record{i} = V_star;
end

% 求每个指标均值和方差
AC(1) = mean(AC_); AC(2) = std(AC_);
NMI(1) = mean(NMI_); NMI(2) = std(NMI_);
purity(1) = mean(purity_); purity(2) = std(purity_);
Fscore(1) = mean(Fscore_); Fscore(2) = std(Fscore_);
Precision(1) = mean(Precision_); Precision(2) = std(Precision_);
Recall(1) = mean(Recall_); Recall(2) = std(Recall_);
AR(1) = mean(AR_); AR(2) = std(AR_);
fprintf(fileID, "alpha = %g, beta = %g:\n", alpha, beta);
fprintf(fileID, "AC = %5.4f + %5.4f, NMI = %5.4f + %5.4f, purity = %5.4f + %5.4f\nFscore = %5.4f + %5.4f, Precision = %5.4f + %5.4f, Recall = %5.4f + %5.4f, AR = %5.4f + %5.4f\n",...
    AC(1), AC(2), NMI(1), NMI(2), purity(1), purity(2), Fscore(1), Fscore(2), Precision(1), Precision(2), Recall(1), Recall(2), AR(1), AR(2));
fprintf(fileID,'*****************************************************************************************************\n');
plot(obj);
% 保存记录矩阵
save_file_name = ['./result/BBCSport_', num2str(int32(AC(1)*10000)), '.mat'];
save(save_file_name, 'res_record', 'obj', 'V_star_record', 'alpha', 'beta');






