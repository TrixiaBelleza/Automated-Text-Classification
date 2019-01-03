def printStatus(res):
	if res.ok:
		print("OK " + str(res.status_code));
	else:
		print("ERROR " + str(res.status_code));