import config, os, sys
import websocket, json

def on_open(ws):
    print("opened")
    auth_data = {"action": "auth","key": config.API_KEY,"secret": config.SECRET_KEY}
    ws.send(json.dumps(auth_data))
    listen_message = {"action": "subscribe","bars": ["AAPL"]}
    ws.send(json.dumps(listen_message))


def on_message(ws, message):
    print(message)
    if(json.loads(message)[0]['S'] == 'AAPL'):
        print('AAPL')
        data = json.loads(message)
        print(data)
        timedata = data[0]['t'][0:10] + '-' + data[0]['t'][11:16]
        closedata = data[0]['c']
        highdata = data[0]['h']
        lowdata = data[0]['l']
        opendata = data[0]['o']
        adjdata = 0
        volumedata = data[0]['v']
        # print(timedata + ',' + closedata +',' +  highdata + ',' + lowdata + ',' + opendata + ',' + adjdata + ',' + volumedata)
        # print(timedata + ',' + str(closedata) + ',' + str(highdata)+ ',' + str(lowdata)+ ',' + str(opendata)+ ',' + str(adjdata) + ',' + str(volumedata))
        testfile =  open("samplefile.txt", "a")
        testfile.write(timedata + ',' + str(closedata) + ',' + str(highdata)+ ',' + str(lowdata)+ ',' + str(opendata)+ ',' + str(adjdata) + ',' + str(volumedata) + '\n')
        testfile.close()
        #print(timedata)
    
    print(message)
    pritn("Done")

    


def on_close(ws):
    print("closed connection")

socket = "wss://stream.data.alpaca.markets/v2/iex"

ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()