import math
from datetime import timedelta, datetime

def _method_parser(instance, method_str : str) -> None:
    '''Execute the method for the instance inside the string'''
    #To avoid using eval() function
    method_str = method_str.split("(")
    method = method_str[0]
    method_args = method_str[1][:-1].split(',')
    method_args_dict = {}
    for arg in method_args:
        arg = arg.split("=")
        arg = [x.strip() for x in arg]
        method_args_dict[arg[0]] = arg[1].strip("'")
    return getattr(instance, method)(**method_args_dict)

def _proper_time_type(time : float, 
                      attribute_type : str, 
                      len_step : int,
                      datetime : datetime = None,
                      next_day : bool = False) -> datetime:
    next_day = int(next_day)
    rounded_minutes = len_step * round(time * 60 / len_step)
    if attribute_type == "datetime_fixed":
        proper_time = [math.floor(rounded_minutes/60), rounded_minutes%60]
        #TODO: keep only this one
        if proper_time[0] == 24:
            proper_time[0] = 0
        return proper_time
    else:
        _minutes = timedelta(minutes = rounded_minutes)
        if attribute_type == "datetime_variable":
            return datetime.replace(day = datetime.day+next_day,
                                    hour = _minutes.seconds//3600, 
                                    minute = (_minutes.seconds//60)%60)
        elif attribute_type == "timedelta":
            return datetime + _minutes

