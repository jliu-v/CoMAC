CUDA_DEVICE=0
# model_name=$2

use_tiny_dataset="_tiny"
train_dataset_path="data/train_focus${use_tiny_dataset}.json"
valid_dataset_path="data/valid_focus${use_tiny_dataset}.json"

for kp_method in focus comac; do
for model_name in GPT2 BART; do
for pg_label_weight in 0.9; do
for pg_loss_sample_p in 0.2; do
for k in 1; do
for p in 1 ; do
    
      l=10

      echo "${kp_method} ${model_name}"

	    
	    
	    flag=${kp_method}_${model_name}_L${l}_K${k}_P${p}_PW${pg_label_weight}_PS${pg_loss_sample_p}
	    model_dir="ncolbert_v6_tfidf_search4/models_PW${pg_label_weight}_PS${pg_loss_sample_p}"
	    log_dir="ncolbert_v6_tfidf_search4/logs_PW${pg_label_weight}_PS${pg_loss_sample_p}"
	    idf_file="data/term_idf_${model_name}.csv"
	    extra_arg=""

            

	    if [ "${pg_label_weight}" != "None" ]; then
	        extra_arg="${extra_arg} --pg_label_weight ${pg_label_weight}"
	    fi

	    if [ "${pg_loss_sample_p}" != "None" ]; then
          extra_arg="${extra_arg} --pg_loss_sample_p ${pg_loss_sample_p}"
      fi


	    mkdir -p ${model_dir}
	    mkdir -p ${log_dir}
            

	    lock_file=${log_dir}/${flag}.lock
	    if [ ! -f "${lock_file}" ]; then
          # create lock file
          touch ${lock_file}

	        # train
	        FILE=${log_dir}/train_${flag}.log
          if [ -f "$FILE" ]; then
            echo "[CUDA_DEVICE=${CUDA_DEVICE}] File $FILE exists."
          else
            echo "[CUDA_DEVICE=${CUDA_DEVICE}] File $FILE not exists. Working on file ${FILE}"
            CUDA_VISIBLE_DEVICES= python train.py \
              --kp_method ${kp_method} \
              --model_name ${model_name} \
              --train_dataset_path ${train_dataset_path} \
              --dev_dataset_path ${valid_dataset_path} \
              --n_epochs 2 \
              --lm_coef ${l} \
              --kn_coef ${k} \
              --ps_coef ${p} \
              --flag ${flag} \
              --incontext \
              --train_batch_size 6 \
              --model_dir ${model_dir} \
              --idf_file ${idf_file} \
              --lr 1e-04 \
              ${extra_arg} > ${FILE} 2>&1
              # perl -pi.bak -e 's//\n/g' ${FILE}
              # rm -r ${FILE}.bak
          fi

	        # test
          FILE=${log_dir}/test_${flag}_MTL.log
          if [ -f "$FILE" ]; then
            echo "[CUDA_DEVICE=${CUDA_DEVICE}] File $FILE exists."
          else
            echo "[CUDA_DEVICE=${CUDA_DEVICE}] File $FILE not exists. Working on file ${FILE}"
            CUDA_VISIBLE_DEVICES= python evaluate_test.py \
              --kp_method ${kp_method} \
              --model_name ${model_name} \
              --test_dataset_path ${valid_dataset_path} \
              --model_checkpoint ${model_dir}/${flag} \
              --idf_file ${idf_file} \
              > ${FILE} 2>&1
              # perl -pi.bak -e 's//\n/g' ${FILE}
              # rm -r ${FILE}.bak
          fi

	        # test ppl
	        FILE=${log_dir}/test_${flag}_MTL_ppl.log
          if [ -f "$FILE" ]; then
            echo "[CUDA_DEVICE=${CUDA_DEVICE}] File $FILE exists."
          else
            echo "[CUDA_DEVICE=${CUDA_DEVICE}] File $FILE not exists. Working on file ${FILE}"
            CUDA_VISIBLE_DEVICES= python evaluate_test_ppl.py \
              --kp_method ${kp_method} \
              --model_name ${model_name} \
              --test_dataset_path ${valid_dataset_path} \
              --model_checkpoint ${model_dir}/${flag} \
              --idf_file ${idf_file} \
              > ${FILE} 2>&1
            # perl -pi.bak -e 's//\n/g' ${FILE}
            # rm -r ${FILE}.bak
          fi

          # remove lock file
          rm ${lock_file}

	    fi	# end: if [ ! -f "${lock_file}" ]

done
done
done
done
done
done