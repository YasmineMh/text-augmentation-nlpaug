import random
import re
from datetime import datetime


date_formats = ["%d-%B-%Y", "%m-%d-%Y", "%m/%d/%Y", "%d-%b-%Y", "%Y-%m-%d", "%b %d %Y", "%b. %d, %Y", "%B. %d, %Y", "%d %B %Y", "British days_numbers", "British days_letters", "American days_numbers", "American days_letters"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
days = [31, 28, 31, 30, 31, 30 , 31, 31, 30, 31, 30, 31]

days_numbers = ["1st", "2nd	", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "21st", "22nd", "23rd", "24th", "25th", "26th", "27th", "28th", "29th", "30th", "31st"]
days_letters = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third", "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first"]


def transform_date_type(old_date, change_month=True, change_day=True):
    """
    Transform the date label into another type of date

    Args:
        old_date: date extracted form the paragraph      
        change_month: change old_date month randomly
        change_day: change old_date day randomly

    Returns:
        New date with different format and values (if change_month=True or/and change_day=True)

    """
    #Select a month randomly
    random_month = random.randint(0, len(months)-1)

    #Select a date format to transform the old date into another type of date
    new_date_format = random.choice(date_formats)

    #Extract the month of the old date
    old_month_re = re.search(r'(\w+) \d+, \d{4}', old_date)
    old_month = old_month_re.group(1)

    #Extract the day of the old date
    old_day_re = re.search(r'\w+ (\d+), \d{4}', old_date)
    old_day = old_day_re.group(1)

    #Extract the year of the old date
    year_re = re.search(r'\w+ \d+, (\d{4})', old_date)
    year = year_re.group(1)

    #Replace the old month and day by random ones
    new_date = old_date
    new_month = old_month
    new_day = old_day
    if change_month:
        new_month = months[random_month]
        ew_date = new_date.replace(old_month, new_month, 1)
    if change_day:
        new_day = str(random.randint(1, days[random_month]))
        new_date = new_date.replace(old_day, new_day, 1)

    #Transform the date
    if new_date_format == "British days_numbers":
        #22nd January 1999
        new_date = str(days_numbers[int(new_day)-1]) + " " + new_month + " " + year
    elif new_date_format == "British days_letters":
        #the Twenty-second of January, 1999	
        new_date = "the " + str(days_letters[int(new_day)-1]).capitalize() + " of " + new_month + ", " + year
    elif new_date_format == "American days_numbers":
        #January 22nd, 1999
        new_date = new_month + " " + str(days_numbers[int(new_day)-1]) + ", " + year
    elif new_date_format == "American days_letters":
        #January the Twenty-second, 1999
        new_date = new_month + " the " + str(days_letters[int(new_day)-1]).capitalize() + ", " + year
    else:
        new_date = datetime.strptime(new_date, '%B %d, %Y').strftime(new_date_format)

    return new_date