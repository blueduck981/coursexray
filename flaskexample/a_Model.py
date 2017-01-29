def ModelIt(fromUser  = 'Default', reviews = [], course_from_sql = []):
#def ModelIt(fromUser  = 'Default', births = []):

  #in_month = len(births)
	keyword_1 = fromUser.split(', ')
	print type(reviews)





	print 'The first keyword is  %s' % keyword_1
	result = keyword_1
  


	if fromUser != 'Default':
		return result
	else:
		return 'check your input'