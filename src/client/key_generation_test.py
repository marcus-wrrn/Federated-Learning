import server.client as client

temp = client.get_key()

print(temp)
temp2 = client.upload_key('http://192.168.2.24:5000/training')
print(temp2)