import json, os
ST = {r["id"]: r for r in json.load(open(os.path.expanduser("~/nmj/sesqpaper/ext2/stats.json")))}
EXP = {"S0035": {"8": 6}, "S0368": {"9": 8}, "S2386": {"10": 12}, "S2385": {"10": 12}, "S2128": {"10": 12},
       "S4112": {"10": 12, "12": 2}, "S6071": {"10": 7, "11": 10}, "S7513": {"11": 24}, "S7520": {"11": 24},
       "S7525": {"11": 24}, "S7524": {"11": 24}, "S7527": {"11": 24, "12": 8}}
for i, e in EXP.items():
    print("OK " if ST[i]["big"] == e else "FAIL", i, ST[i]["big"], ST[i]["F"], ST[i]["inF"])
