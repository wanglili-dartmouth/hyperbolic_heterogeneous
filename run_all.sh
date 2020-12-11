#!/bin/bash


e=5
exp=nc_experiment

for dataset in movie
do
     for dim in  25 20 15 10 5 2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/nr/${dataset}/dim=${dim}

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/md" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python nr_md_movie.py ${args}	     
     
done
done

for dataset in movie
do
     for dim in  25 20 15 10 5 2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/nr/${dataset}/dim=${dim}

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/ma" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python nr_ma_movie.py ${args}	     
     
done
done



for dataset in dblp
do
     for dim in  25 20 15 10 5 2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/nr/${dataset}/dim=${dim}

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/pa" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python nr_pa_dblp.py ${args}	     
     
done
done


for dataset in dblp
do
     for dim in  25 20 15 10 5 2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/nr/${dataset}/dim=${dim}

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/pc" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python nr_pc_dblp.py ${args}	     
     
done
done

exp=lp_experiment


for dataset in dblp
do
     for dim in  25 20 15 10 5 2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/lp/${dataset}/dim=${dim}/pc

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/pc" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python lp_pc_dblp.py ${args}	     
     
done
done


for dataset in dblp
do
     for dim in  25 20 15 10 5 2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/lp/${dataset}/dim=${dim}/pa

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/pa" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python lp_pa_dblp.py ${args}	     
     
done
done


for dataset in movie
do
     for dim in  25 20 15 10 5 2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/lp/${dataset}/dim=${dim}/ma

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/ma" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python lp_ma_movie.py ${args}	     
     
done
done

for dataset in movie
do
     for dim in  25 20 15 10 5 2
     do
            edgelist=data/${dataset}.edgelist
            embedding_dir=embeddings/lp/${dataset}/dim=${dim}/md

            test_results=$(printf "test_results/${dataset}/${exp}/dim=%03d/md" ${dim})
            embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz"  ${e})
            echo $embedding_f
            
            args=$(echo --edgelist ${edgelist} --embedding ${embedding_dir} --dim ${dim}  -e ${e} --test-results-dir ${test_results} )  
            python lp_md_movie.py ${args}	     
     
done
done

