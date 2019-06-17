#!/bin/bash -e 

cd /primitives
git branch rffeatures_pipelines
git checkout rffeatures_pipelines
#git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
#git pull upstream master

Datasets=('1491_one_hundred_plants_margin' 'LL0_1100_popularkids')
cd /src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/
mkdir test_pipeline
cd test_pipeline

# create text file to record scores and timing information
touch scores.txt
echo "DATASET, F1 SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  # generate and save pipeline + metafile
  python3 "/src/rffeaturesd3mwrapper/RffeaturesD3MWrapper/python_pipeline_generator_rffeatures.py" $i

  # test and score pipeline
  start=`date +%s` 
  python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
  end=`date +%s`
  runtime=$((end-start))

  # copy pipeline if execution time is less than one hour
  if [ $runtime -lt 3600 ]; then
     echo "$i took less than 1 hour, copying pipeline"
     cp * /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.rffeatures.Rffeatures/3.1.1/pipelines
  fi

  # save information
  IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  echo "$i, $score, $runtime" >> scores.txt
  
  # cleanup temporary file
  rm *.meta
  rm *.json
  rm scores.csv
done