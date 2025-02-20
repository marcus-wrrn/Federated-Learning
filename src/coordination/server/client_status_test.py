from flask import Flask, render_template
app = Flask(__name__)

client_data = (
    (0,"12EMA36S","192.168.32.2",1,2),
    (1,"S0JJTJVE","192.128.64.32",1,2),
    (2,"MOIEKSAS","256.128.64.32",1,2),
    (3,"KYAVP4R9","192.155.0.1",1,2),
)

@app.route("/client_status")
def get_status():
    return render_template("client_table.html",data=client_data)

data1 = (
    ("12EMA36S","PATH/12EMA36S.pth",1,0.85),
    ("S0JJTJVE","PATH/S0JJTJVE.pth",1,0.3),
    ("MOIEKSAS","PATH/MOIEKSAS.pth",1,0.9),
    ("KYAVP4R9","PATH/KYAVP4R9.pth",1,0.8),
    ("Round1","PATH/Round1.pth",1,0.65),
)

data2 = (
    ("12EMA36S","PATH/12EMA36S.pth",1,0.87),
    ("S0JJTJVE","PATH/S0JJTJVE.pth",-1,0.3),
    ("MOIEKSAS","PATH/MOIEKSAS.pth",0,"--"),
    ("KYAVP4R9","PATH/KYAVP4R9.pth",1,0.6),
    ("Round2","--",0,"--"),
)

@app.route("/training_status")
def get_training_status():
    return render_template("training_table.html",data=data1,round_data2=data2)


if __name__ == "__main__":    
    print("Running client list test")
    app.run()