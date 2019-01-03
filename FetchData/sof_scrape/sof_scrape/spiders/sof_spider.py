import scrapy

class QuotesSpider(scrapy.Spider):
	name = "quotes"

	def start_requests(self):
		urls = [
			'https://stackoverflow.com/questions/ask?wizard=1'
		]
		for url in urls:
			yield scrapy.Request(url=url, callback=self.parse)

	def parse(self, response):
		
		page = response.xpath('//div[@class = "container"]')
		print("Page")
		print(page)

		for p in page:

			print("p")
			print(p)
			print("End of p")
			# print("Quote:")
		# print(page)
		# filename = 'quotes-%s.html' % page
		# with open(filename,'wb') as f:
			# f.write(response.body)
		# self.log('Saved file %s' % filename)