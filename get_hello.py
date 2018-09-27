import urllib.request
import time
import json
data2={'geg':1,"gegg":2}
post_data = json.dumps(data2)
header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
               'Content-Type': 'application/json'}
url_all="http://127.0.0.1:5000/ana"
req = urllib.request.Request(url=url_all, data=bytes(post_data, encoding="utf8"), headers=header_dict)
res = urllib.request.urlopen(req).read()
print(res)