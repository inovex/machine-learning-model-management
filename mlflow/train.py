from utils import *
import git
def logMlflow(model,data,param=dict(),metrics=dict(),features=None, tags=dict(),run_name=None):
    # Imports
    import mlflow
    import os
    from sklearn.externals import joblib
    
    # Get some general information
    output_folder = "mlflow_out"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    type = model.__module__.split(".")[0]
    modelname = model.__class__.__name__
    
    # Start actual logging
    
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    mlflow.set_experiment(experiment_name="demo")
    if not run_name:
        run_name = modelname
    with mlflow.start_run(source_name=repo.remotes.origin.url,source_version=sha, run_name=run_name):
        
        # Log Parameters
        for k,v in param.items():
            mlflow.log_param(k, v)

        # Track dependencies
        import pkg_resources
        with open("{}/dependencies.txt".format(output_folder), "w+") as f: 
            for d in pkg_resources.working_set:
                f.write("{}={}\n".format(d.project_name,d.version))
        mlflow.log_artifact("{}/dependencies.txt".format(output_folder))
        
        # Track data
        data.to_csv("{}/data".format(output_folder))
        mlflow.log_artifact("{}/data".format(output_folder))
        
        if type=="sklearn":
            _ = joblib.dump(model,"{}/sklearn".format(output_folder))
            mlflow.log_artifact("{}/sklearn".format(output_folder))
        if type=="lgb":
            model.save_model("{}/lghtgbm.txt".format(output_folder))
            mlflow.log_artifact("{}/lghtgbm.txt".format(output_folder))
        
        # Log metrics
        for k,v in metrics.items():
            mlflow.log_metric(k,v)

        # plot Feature importances if avaible
        featurePlot = plotFeatureImportances(model, features, type)
        if featurePlot:
            mlflow.log_artifact("{}.png".format(featurePlot))
            
        # Set some tags to identify the experiment
        mlflow.set_tag("model",modelname)
        for tag, v in tags.items():
            mlflow.set_tag(t,v)
            
def run(clf, params, run_name=None):
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    metrics = eval_metrics(y_test, predictions)
    print(metrics['mae'], metrics['r2'])

    logMlflow(clf,data,param=params,metrics=metrics,features=x_test.columns.values, run_name=run_name)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNet
    import sys
    # Do a train_test_split
    data = getData()
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=10, random_state=42)
    
    params=dict(alpha=float(sys.argv[1]) if len(sys.argv) > 1 else 0.5,
                l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)

    clf = ElasticNet(**params)

    run(clf,params)
