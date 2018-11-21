from pdf2image import convert_from_path

def convert2images(file_loc):
	try:
		pages = convert_from_path(file_loc,500)
		val = 1
		for page in pages:
			filename = "page_"+str(val)+".png"
			page.save(filename,"PNG")
			val = val + 1
	except Eception as error :
		print ("[-] ",error)

# File Loc = 'sentence_dataset.pdf'