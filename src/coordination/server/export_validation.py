import validation as val

from flask import Flask, render_template
app = Flask(__name__)


# Takes in a model, and passes it to validation
# export 
# ability  to choose the model you want 
def get_data(dataset,model):
    temp = val.validation("cpu",dataset,model)
    accuracy = temp

# Export Resutls
@app.route("/validation_resutls")
def print_validation():
    # get the validation results and export them to a table 
    accuracy,recall,precision,f1 = get_data()
    return render_template("validation_table.html",accuracy=data1,recall=data4,precision=data2,f1=data3)

# Get model 
@app.route("/validation_list")
def get_validation_model():
    # This gets all the models from the database
    # puts them in a drop down
    # User can select a model
    # then they hit a button and it will load the results 
    # This could also have the ability to load the results from a database, to provide some additional info, or quick refernce 

    return render_template("training_table.html",data=data1,round_data2=data2)
