def time_fun(x):
    days = x // 86400
    hours = x % 86400 // 3600
    minutes = x % 86400 % 3600 // 60
    seconds = x % 86400 % 3600 % 60
    print('预计完成时间为：{}d {}h {}m {}s'.format(days, hours, minutes, seconds))
