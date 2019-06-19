#!/bin/bash -e
rm /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.rffeatures.Rffeatures/3.1.1/pipelines/test_pipeline/*.meta
rm /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.rffeatures.Rffeatures/3.1.1/pipelines/test_pipeline/*.json
#mkdir /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.rffeatures.Rffeatures/3.1.1/pipelines/test_pipeline
touch /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.rffeatures.Rffeatures/3.1.1/pipelines/experiments/dum.txt
cd /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.rffeatures.Rffeatures/3.1.1/pipelines/test_pipeline
best_score=0
match="step_1.add_output('produce')"
#insert="Temporary Line num_features"
file="/src/rffeaturesd3mwrapper/RffeaturesD3MWrapper/python_pipeline_generator_rffeatures.py"
#sed -i "s/$match/$match\n$insert/" $file
for n in $(seq 3 3 63); do
  sed -i '/num_features/d' $file
  insert="step_1.add_hyperparameter(name='num_features', argument_type=ArgumentType.VALUE,data=$n)"
  sed -i "s/$match/$match\n$insert/" $file
  # generate and save pipeline + metafile
  python3 "/src/rffeaturesd3mwrapper/RffeaturesD3MWrapper/python_pipeline_generator_rffeatures.py" '1491_one_hundred_plants_margin'

  # test and score pipeline
  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
  end=`date +%s`
  runtime=$((end-start))

  # copy pipeline if execution time is less than one hour
  if [ $runtime -lt 3600 ]; then
    echo "$i took less than 1 hour, evaluating pipeline"
    IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
    echo "$score"
    echo "$best_score"
    if [[ $score > $best_score ]]; then
      echo "$n, $score, $runtime" >> scores.txt
      best_score=$score
      echo "$best_score"
      rm ../experiments/*
      cp *.meta ../experiments/
      cp *.json ../experiments/
    fi
  fi
 
  #cleanup temporary file
  rm *.meta
  rm *.json
  rm scores.csv
done
