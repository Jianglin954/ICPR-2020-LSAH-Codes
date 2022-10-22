clear,clc; close;
warning('off')
load ./datasets/cifar_10_gist
dataset = 'cifar_10_gist'; AnchorNum = 500; Knum=5; maxItr=10; alpha=10;
display([dataset ': ']);
traindata = double(traindata);
testdata = double(testdata);
nbits = [8, 16, 32, 64, 96, 128];  

%% run algo
method = 'LSAH';

display([method ': ']);
    for i=1:length(nbits)
        t1=clock;
        switch method
            case 'LSAH'
                ek=clock;
                anchor_nm = ['anchor_' num2str(AnchorNum)];
                eval(['[~,' anchor_nm '] = litekmeans(traindata, AnchorNum, ''MaxIter'', 10);']);
                eval(['anchor_set.' anchor_nm '= ' anchor_nm, ';']);
                eval(['anchor = anchor_set.' anchor_nm ';']);
                ek2=clock;
                kmeantime=etime(ek2,ek)
                etrain=clock;
                desireddim=nbits(i); 
                [ Z ] = constructAGHW( traindata, anchor, Knum ,0 ); 
%                 [~,Z, ~] = get_Z(traindata, anchor, Knum, 0); 
                display('training...');        
                [P,G,W,Banchor,~] = LSAH(traindata,Z,25,nbits(i),10^-2,10^3,10^0,10);    
                H = traindata*P > 0;
                etrain2=clock;
                traintime2=etime(etrain2,etrain)

                etest=clock;
                tH = testdata*P > 0;
                etest2=clock;
                testtime=etime(etest2,etest)
        end
        t2=clock;
        traintime(i)=etime(t2,t1)

        %% evaluation
        display('Evaluation...');
        hammRadius = 2;
        B = compactbit(H);
        tB = compactbit(tH);
        clear H tH;    
        hammTrainTest = hammingDist(tB, B)';
        Ret = (hammTrainTest <= hammRadius+0.00001);
        [Pre(i), Rec(i)] = evaluate_macro(cateTrainTest, Ret)
        Fmeasure(i) = F1_measure(Pre(i), Rec(i))
        [~, HammingRank]=sort(hammTrainTest,1);
        MAP(i) = cat_apcalNew(cateTrainTest, HammingRank)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
save(['results/',method, 'test'],'MAP','Pre','Rec','Fmeasure');
