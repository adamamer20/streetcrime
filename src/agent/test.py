import mesa_geo
import shapely

class Mover:
    attributes = {"mover_attr": 1}
    
    def __init__(self):
        self_classes = self.__class__.__mro__[1:-1]
        for class_obj in self_classes:
            self.attributes.update(class_obj.attributes)
        
    def gen_attribute(self, attribute_name : str, attribute_type : str = "float", distribution = "normal") -> float:
        """Changes the self.data[f'{attribute_name}'] to a random value for the next activity based on 
        the mean, the sd and the limits specified in params

        Arguments:
            attribute_name -- can be "self_defence", "crime_motivation"
        """
        match attribute_type:
            case "float":
                limits = [0,1]
            case "datetime_fixed", "datetime_variable":
                limits = [float('-inf'),float('+inf')]
                def __next_day(time : float) -> int:
                    if time >= 24 or attribute_type == "resting_end":
                        return 1
                    else:
                        return 0
                def __adjust_time(time : float) -> datetime:
                    rounded_minutes = self.model.model_params['len_step'] * round(float_time * 60 / self.model.model_params['len_step'])
                    if attribute_type == "datetime_fixed":
                        adj_time = [math.floor(rounded_minutes/60), rounded_minutes%60]
                    else:
                        _minutes = timedelta(minutes = rounded_minutes)
                        adj_time = self.model.data['datetime'].replace(hour = _minutes.seconds//3600, minute = (_minutes.seconds//60)%60) + timedelta(days = next_day)
        mean = self.params[f'mean_{attribute_name}']
        sd = self.params[f'sd_{attribute_name}']
        try: 
            limits[0] = self.params[f'min_{attribute_name}']
        except KeyError:
            pass
        try: 
            limits[1] = self.params[f'max_{attribute_name}']
        except KeyError:
            pass
        attribute_value = np.random.normal(mean, sd)
        if attribute_type == "datetime":
            next_day = __next_day(attribute_value)
        while attribute_value < limits[0] or attribute_value > limits[1]:
            attribute_value = np.random.normal(mean, sd)
            if attribute_type == "datetime":
                next_day = __next_day(attribute_value)
        if attribute_type == "datetime":
            return __adjust_time(attribute_value)
        else:
            return attribute_value
        
class Resident(Mover):
    attributes = {"resident_attr": 2}

    def __init__(self):
        super().__init__()

class Worker(Resident):
    attributes = {"worker_attr": 3}

    def __init__(self):
        super().__init__()


agentcreator = mesa_geo.AgentCreator(Worker, crs = 'espg:7991', agent_kwargs={"params": {"mean_worker_attr": 1}})

agent = agentcreator.create_agent(geometry = shapely.Point(10,10), unique_id = 1)

print(agent)