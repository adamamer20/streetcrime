import unittest
import geopandas as gpd
import osmnx as ox
from streetcrime.model import StreetCrime

from streetcrime.space.city import City
from streetcrime.agents.mover import Mover
from streetcrime.agents.informed_mover import InformedMover
from streetcrime.agents.resident import Resident
from streetcrime.agents.criminal import Pickpocket, Robber
from streetcrime.agents.worker import Worker
from streetcrime.agents.police_agent import PoliceAgent
import numpy as np

from datetime import datetime, timedelta

class TestModel(unittest.TestCase):
    model = None
    initialized = False

    @classmethod
    def setUpClass(cls):
        if not cls.initialized:
            cls.model = StreetCrime(
                space = City(
                        crs = 'epsg:7791', 
                        roads = 'tests/data/processed/roads.gpkg',
                        public_transport = 'tests/data/processed/public_transport.gpkg',
                        neighborhoods = 'tests/data/processed/neighborhoods.gpkg',
                        buildings = 'tests/data/processed/buildings.shp'),
                p_agents=None,
                len_step=3)
            cls.initialized = True
        else:
            pass
        
class TestMover(TestModel):
    
    def setUp(self) -> None:
        self.mover = Mover(unique_id = 1,
                           model = self.model,
                           geometry = self.model.space.buildings.geometry.iloc[0].centroid,
                           crs = 'epsg:7991')
    
    def test_mover_step(self):
        self.mover.step()
        self.assertEqual(self.mover.status, 'transport')
        
    def test_mover_complete_path(self):
        while self.mover.status != 'busy':
            destination_id = self.mover.destination['id']
            self.mover.step()
        self.assertEqual(self.mover.geometry, 
                         self.model.space.buildings.loc[destination_id].geometry.centroid)

class TestInformedMover(TestModel):
    
    def setUp(self):
        self.informed_mover = InformedMover(unique_id = 1,
                                            model = self.model,
                                            geometry = self.model.space.buildings.geometry.iloc[0].centroid,
                                            crs = 'epsg:7991',
                                            p_information = 1)
    
    def test_informed_mover_update_info(self):
        self.informed_mover.update_info(info_type = ['crimes', 'visits'])
        self.assertEqual(self.informed_mover.model.info_neighborhoods.shape, (1, 1))
        
class TestResident(TestModel):
    
    def setUp(self):
        self.resident = Resident(unique_id = 1,
                                model = self.model,
                                geometry = self.model.space.buildings.geometry.iloc[0].centroid,
                                crs = 'epsg:7991',
                                p_information = 1)
    
    def test_resident_init(self):
        for time in self.resident.resting_time:
            self.assertIsInstance(time, datetime)
        self.assertIsInstance(self.resident.resting_time[0], datetime)
        self.assertIsInstance(self.resident.resting_time[1], datetime)
        self.assertEqual(self.resident.resting_time[0].day, self.model.datetime.day)
        self.assertEqual(self.resident.resting_time[1].day, self.model.datetime.day+1)
        self.assertIn(self.resident.home, self.model.space.buildings.index)
        self.assertEqual(self.resident.status, "home")
        self.assertIsInstance(self.resident.income, float)
        self.assertGreaterEqual(self.resident.income, 0)
        self.assertIsInstance(self.resident.car, np.bool_)
        
    def test_resident_step(self):
        #test if it gets an activity after resting
        self.resident.resting_time[1] = self.model.datetime + timedelta(minutes = 1)
        self.resident.step()
        self.assertEqual(self.resident.status, 'transport')
        while self.resident.status != 'busy':
            self.resident.step()
        
        #test if the resident goes home when it is resting time
        self.status = 'free'
        self.resident.resting_time[0] = self.model.datetime + timedelta(minutes = -1)
        self.resident.step()
        self.assertEqual(self.resident.status, 'transport')
        while self.resident.status != 'home':
            self.resident.step()
        
        #test if the resting time is generated at 2 pm
        self.model.datetime = self.model.datetime.replace(hour = 14, minute = 0)
        previous_resting_time = self.resident.resting_time
        self.resident.step()
        self.assertNotEqual(self.resident.resting_time, previous_resting_time)
        self.assertIsInstance(self.resident.resting_time[0], datetime)
        self.assertIsInstance(self.resident.resting_time[1], datetime)
        self.assertEqual(self.resident.resting_time[0].day, self.model.datetime.day)
        self.assertEqual(self.resident.resting_time[1].day, self.model.datetime.day+1)

class TestWorker(TestModel):
    def setUp(self) -> None:
        self.worker = Worker(unique_id = 1,
                            model = self.model,
                            geometry = self.model.space.buildings.geometry.iloc[0].centroid,
                            crs = 'epsg:7991')
    
    def test_worker_init(self):
        self.assertGreaterEqual(self.worker.resting_time[0].hour, self.worker.work_time[1][0])
        self.assertLessEqual(self.worker.resting_time[1].hour, self.worker.work_time[0][1])
        self.assertIn(self.worker.work, self.model.space.buildings.index)
        self.assertGreaterEqual(self.worker.defence, 0)
        self.assertLessEqual(self.worker.defence, 1)
        self.assertGreaterEqual(self.worker.crime_attractiveness, 0)
        self.assertLessEqual(self.worker.crime_attractiveness, 1)
    
    def test_worker_step(self):
        #Test if it goes to work when it is work time
        self.model.datetime = self.model.datetime.replace(hour = self.worker.work_time[0][0], 
                                                          minute = self.worker.work_time[0][1])
        self.worker.step()
        self.assertEqual(self.worker.status, 'transport')
        while self.worker.status != 'work':
            self.worker.step()
        self.assertEqual(self.worker.geometry, 
                         self.model.space.buildings.loc[self.worker.work].geometry.centroid)

        #Test if the worker goes away from work when it is end of work time
        self.model.datetime = self.model.datetime.replace(hour = self.worker.work_time[1][0], 
                                                          minute = self.worker.work_time[1][1]+1)
        self.worker.step()
        self.assertIn(self.worker.status, ['transport', 'free'])

class TestCriminal(TestModel):
    def setUp(self) -> None:
        self.pickpocket = Pickpocket(unique_id = 1,
                                model = self.model,
                                geometry = self.model.space.buildings.geometry.iloc[0].centroid,
                                crs = 'epsg:7991')
        self.pickpocket.geometry = self.model.space.buildings.geometry.iloc[0].centroid
        self.pickpocket.crime_motivation = 0.9
        self.robber = Robber(unique_id = 2,
                                model = self.model,
                                geometry = self.model.space.buildings.geometry.iloc[0].centroid,
                                crs = 'epsg:7991')
        self.robber.geometry = self.model.space.buildings.geometry.iloc[0].centroid
        self.robber.crime_motivation = 0.9
        self.worker1 = Worker(unique_id = 3,
            model = self.model,
            geometry = self.model.space.buildings.geometry.iloc[0].centroid,
            crs = 'epsg:7991')
        self.worker1.geometry = self.model.space.buildings.geometry.iloc[0].centroid
        self.worker1.crime_attractiveness = 0.1
        self.worker1.defence = 0
        self.worker1.status = 'transport'
        self.model.space.add_agents([self.pickpocket, self.robber, self.worker1])

    
    def test_pickpocket_commit_crime(self):
        #Test if it correctly commits a crime
        self.pickpocket.commit_crime()
        
        self.assertEqual(self.model.crimes.iloc[-1].datetime, self.model.datetime)
        self.assertEqual(self.model.crimes.iloc[-1].position, self.pickpocket.geometry)
        self.assertEqual(self.model.crimes.iloc[-1].criminal, self.pickpocket.unique_id)
        self.assertEqual(self.model.crimes.iloc[-1].witnesses, 0)
        self.assertEqual(self.model.crimes.iloc[-1].victim, self.worker1.unique_id)
        self.assertEqual(self.model.crimes.iloc[-1].type, 'pickpocketing')
        self.asserTrue(self.model.crimes.iloc[-1].successful)
        
        #Test if it correctly chooses victim
        self.worker2 = Worker(unique_id = 4,
            model = self.model,
            geometry = self.model.space.buildings.geometry.iloc[0].centroid,
            crs = 'epsg:7991')
        self.worker2.geometry = self.model.space.buildings.geometry.iloc[0].centroid
        self.worker2.crime_attractiveness = 1
        self.worker2.defence = 1
        self.worker2.status = 'transport'
        self.model.space.add_agents([self.worker2])
        
        self.pickpocket.commit_crime()
        
        self.assertEqual(self.model.crimes.iloc[-1].witnesses, 1)
        self.assertEqual(self.model.crimes.iloc[-1].victim, self.worker2.unique_id)
        self.assertEqual(self.model.crimes.iloc[-1].type, 'pickpocketing')
        self.assertFalse(self.model.crimes.iloc[-1].successful)
    
    def test_robber_commit_crime(self):
        #Test if it correctly commits a crime
        self.robber.commit_crime()
        
        self.assertEqual(self.model.crimes.iloc[-1].datetime, self.model.datetime)
        self.assertEqual(self.model.crimes.iloc[-1].position, self.robber.geometry)
        self.assertEqual(self.model.crimes.iloc[-1].criminal, self.robber.unique_id)
        self.assertEqual(self.model.crimes.iloc[-1].witnesses, 0)
        self.assertEqual(self.model.crimes.iloc[-1].victim, self.worker1.unique_id)
        self.assertEqual(self.model.crimes.iloc[-1].type, 'robbery')
        self.assertTrue(self.model.crimes.iloc[-1].successful)
        
        #Test if it correctly chooses victim
        self.worker2 = Worker(unique_id = 4,
            model = self.model,
            geometry = self.model.space.buildings.geometry.iloc[0].centroid,
            crs = 'epsg:7991')
        self.worker2.geometry = self.model.space.buildings.geometry.iloc[0].centroid
        self.worker2.crime_attractiveness = 1
        self.worker2.defence = 1
        self.worker2.status = 'transport'
        self.model.space.add_agents([self.worker2])
        
        self.robber.commit_crime()
        
        self.assertEqual(self.model.crimes.iloc[-1].witnesses, 1)
        self.assertEqual(self.model.crimes.iloc[-1].victim, self.worker2.unique_id)
        self.assertEqual(self.model.crimes.iloc[-1].type, 'robbery')
        self.assertEqual(self.model.crimes.iloc[-1].successful, False)
    
if __name__ == '__main__':
    unittest.main()