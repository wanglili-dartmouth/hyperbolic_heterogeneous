e=5
for dataset in dblp
do
     for dim in  2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/nr/${dataset}/dim=${dim}

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/pc" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python vis.py ${args}	     
     
done
done