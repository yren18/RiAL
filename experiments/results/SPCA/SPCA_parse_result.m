% Result1

parse_input_result(Result1, 1000, 10, 0.50);
parse_input_result(Result2, 1000, 10, 0.50);
parse_input_result(Result3, 1000, 10, 0.50);
parse_input_result(Result4, 1000, 10, 0.50);
parse_input_result(Result5, 1000, 10, 0.50);
parse_input_result(Result6, 1000, 10, 0.50);
parse_input_result(Result7, 1000, 10, 0.50);
parse_input_result(Result8, 1000, 10, 0.50);
parse_input_result(Result9, 1000, 10, 0.50);
parse_input_result(Result10, 1000, 10, 0.50);
parse_input_result(Result11, 1000, 10, 0.50);
parse_input_result(Result12, 1000, 10, 0.50);
parse_input_result(Result13, 1000, 10, 0.50);


% d = 1000;
% mu = 3.00;
% r = 
% 
% generate_table_text(Result1, 1000, 10, 0.50)
% generate_table_text(Result2, 1000, 10, 0.75)
% generate_table_text(Result3, 1000, 10, 1.00)
% generate_table_text(Result4, 1000, 10, 1.25)
% generate_table_text(Result5, 1000, 10, 1.50)
% generate_table_text(Result6, 2000, 4 , 3.00)
% generate_table_text(Result7, 2000, 6 , 3.00)
% generate_table_text(Result8, 2000, 8 , 3.00)
% generate_table_text(Result9, 2000, 10, 3.00)
% generate_table_text(Result10,2000, 12, 3.00)

% mu = 1.50;
% 
% generate_table_text(Result1, 10000, 50, mu);
% generate_table_text(Result2, 20000, 50, mu);
% 
% generate_table_text(Result1, 10000, 50, mu);
% parse_input_result(Result4,    200, 20, 0.10);

% parse_input_result(Result1,    200, 20, 0.10);
% generate_table_text(Result1,   200, 20, 0.10);
% generate_table_text(Result2,   500, 20, 0.10);
% generate_table_text(Result3,  1000, 20, 0.10);
% generate_table_text(Result4,  1500, 20, 0.10);
% generate_table_text(Result5,  2000, 20, 0.10);
% generate_table_text(Result6,  1000, 10, 0.10);
% generate_table_text(Result7,  1000, 15, 0.10);
% generate_table_text(Result8,  1000, 25, 0.10);
% generate_table_text(Result9,  1000, 35, 0.10);
% generate_table_text(Result10, 1000, 20, 0.05);
% generate_table_text(Result11, 1000, 20, 0.15);
% generate_table_text(Result12, 1000, 20, 0.20);
% generate_table_text(Result13, 1000, 20, 0.25);

% parse_input_result(Result2, d, r, mu);
% parse_input_result(Result3, 10000, 50, mu);
% parse_input_result(Result4, 20000, 50, mu);
% parse_input_result(Result5, d, r, mu);
% parse_input_result(Result6, d, r, mu);
% parse_input_result(Result7, d, r, mu);
% parse_input_result(Result8, d, r, mu);
% parse_input_result(Result9, d, r, mu);
% parse_input_result(Result10, d, r, mu);

% parse_input_result(Result11, d, r, mu);
% parse_input_result(Result12, d, r, mu);
% parse_input_result(Result13, d, r, mu);
% parse_input_result(Result14, d, r, mu);
% parse_input_result(Result15, d, r, mu);
% parse_input_result(Result16, d, r, mu);
% parse_input_result(Result17, d, r, mu);
% parse_input_result(Result18, d, r, mu);
% parse_input_result(Result19, d, r, mu);
% parse_input_result(Result20, d, r, mu);
% parse_input_result(Result21, d, r, mu);
% parse_input_result(Result22, d, r, mu);
% parse_input_result(Result23, d, r, mu);
% parse_input_result(Result24, d, r, mu);
% parse_input_result(Result25, d, r, mu);
% parse_input_result(Result26, d, r, mu);
% parse_input_result(Result27, d, r, mu);
% parse_input_result(Result28, d, r, mu);
% parse_input_result(Result29, d, r, mu);
% parse_input_result(Result30, d, r, mu);

function parse_input_result(Result1, d, r, mu)
    fprintf(1, '\n\n=========== Summary: n = %d, r = %d, mu = %.3f==========\n', d, r, mu);
    fprintf(1, 'LS-I:     time = %.3fs, sparsity = %.2f, loss = %.4f, iter = %.0f, total = %.0f \n', Result1(4,2), Result1(5,2), Result1(1,2), Result1(3,2), Result1(6,2));
    fprintf(1, 'LS-II:    time = %.3fs, sparsity = %.2f, loss = %.4f, iter = %.0f, total = %.0f \n', Result1(4,3), Result1(5,3), Result1(1,3), Result1(3,3), Result1(6,3));
    fprintf(1, 'RiALSD:   time = %.3fs, sparsity = %.2f, loss = %.4f, iter = %.0f, total = %.0f \n', Result1(4,4), Result1(5,4), Result1(1,4), Result1(3,4), Result1(6,4));
    fprintf(1, 'RADA-RGD: time = %.3fs, sparsity = %.2f, loss = %.4f, iter = %.0f, total = %.0f \n', Result1(4,1), Result1(5,1), Result1(1,1), Result1(3,1), Result1(6,1));
    fprintf(1, 'RADA-PGD: time = %.3fs, sparsity = %.2f, loss = %.4f, iter = %.0f, total = %.0f \n', Result1(4,6), Result1(5,6), Result1(1,6), Result1(3,6), Result1(6,6));
    fprintf(1, 'ManALM:   time = %.3fs, sparsity = %.2f, loss = %.4f, iter = %.0f, total = %.0f \n', Result1(4,5), Result1(5,5), Result1(1,5), Result1(3,5), Result1(6,5));
end

function generate_table_text(Result1, d, r, mu)        
        RADA_result = Result1(:,1);
        LS1_result = Result1(:,2);
        LS2_result = Result1(:,3);
        RiAL_SD_result = Result1(:,4);
        ManALM_result = Result1(:,5);
        PGD_result = Result1(:,6);

        fprintf("n: %d, r: %d\n===============\n", d, r);
        % fprintf("$\\mu$ & \\multicolumn{23}{c}{$d=%d, N=50, r=%d$}\\\\\n",d,r);
        fprintf("$\\mu$ & \\multicolumn{28}{c}{$d=%d, N=50, r=%d$}\\\\\n",d,r);
        fprintf("\\hline\n");      
        fprintf(['  %.0f & ' ...
            '%.2f & %.1f & %.2f & %.0f & %.0f & ' ...
            '%.2f & %.1f & %.2f & %.0f & %.0f & ' ...
            '%.2f & %.1f & %.2f & %.0f & %.0f & ' ...
            '%.2f & %.1f & %.2f & %.0f & %.0f & ' ...
            '%.2f & %.1f & %.2f & %.0f & ' ...
            '%.2f & %.1f & %.2f & %.0f \\\\ \n'], ...
            r, ...
            RiAL_SD_result(1), RiAL_SD_result(5),RiAL_SD_result(4),RiAL_SD_result(3),RiAL_SD_result(6),...
            ManALM_result(1), ManALM_result(5),ManALM_result(4),ManALM_result(3),ManALM_result(6),...
            RADA_result(1), RADA_result(5),RADA_result(4),RADA_result(3),10*RADA_result(3),...
            PGD_result(1), PGD_result(5),PGD_result(4),PGD_result(3),10*PGD_result(3),...
            LS1_result(1), LS1_result(5),LS1_result(4),LS1_result(3),...
            LS2_result(1), LS2_result(5),LS2_result(4),LS2_result(3) ...
            );

        % fprintf(['  %.0f & ' ...
        %     '%.2f & %.1f & %.2f & %.0f & %.0f & ' ...
        %     '%.2f & %.1f & %.2f & %.0f & %.0f & ' ...
        %     '%.2f & %.1f & %.2f & %.0f \\\\ \n'], ...
        %     r, ...
        %     RiAL_SD_result(1), RiAL_SD_result(5),RiAL_SD_result(4),RiAL_SD_result(3),RiAL_SD_result(6),...
        %     RADA_result(1), RADA_result(5),RADA_result(4),RADA_result(3),10*RADA_result(3),...
        %     LS2_result(1), LS2_result(5),LS2_result(4),LS2_result(3) ...
        %     );

        fprintf("\\hline\n");
end
