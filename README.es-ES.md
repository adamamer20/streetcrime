

# streetcrime: un paquete de Python para modelado basado en agentes (ABM) del crimen urbano

El crimen urbano, con sus complejas interacciones locales y dinámicas espaciales, es un excelente candidato para el Modelado Basado en Agentes (ABM). El paquete `streetcrime` está diseñado como una herramienta ABM robusta y fácil de usar para el análisis del crimen urbano. Facilita la prueba de varias teorías del crimen, las compara con datos reales y evalúa estrategias de reducción del crimen. Las versiones futuras incluirán características de Aprendizaje Automático (ML) para la calibración de parámetros. El paquete integra [`osmnx`](https://github.com/gboeing/osmnx) para datos urbanos y [`mesa`](https://github.com/projectmesa/mesa) (junto con[` `mesa-frames`](https://github.com/adamamer20/mesa-frames)) para el marco de modelado.

`streetcrime` fue desarrollado para mi tesis de licenciatura (BSc). Puede encontrar el documento [aquí](https://github.com/adamamer20/streetcrime/blob/main/docs/StreetCrime__A_generative_science_approach.pdf).

## Instalación

### Prerrequisitos
Antes de instalar `streetcrime`, asegúrese de que mesa-frames esté instalado. Dado que aún no está disponible en PyPi ni en conda-forge, siga las instrucciones de instalación [aquí](https://github.com/adamamer20/mesa-frames#installation).

### Pasos de instalación
1. **Clonar el repositorio de GitHub**
    ```bash
    git clone https://github.com/adamamer20/streetcrime.git
    cd streetcrime
    ```

2. **Instalación**
   
   a. *Instalación en un entorno Conda*
      ```bash
      conda activate myenv
      pip install -e .
      ```

   b. *Instalación en un entorno virtual de Python*
      ```bash
      source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
      pip install -e .
      ```

### Uso
*Nota: `streetcrime` está en etapas tempranas de desarrollo; espere cambios y posibles actualizaciones que rompan la compatibilidad. Se agradecen los comentarios y los reportes de problemas.*

- Puede encontrar la documentación de la API [aquí](https://adamamer20.github.io/streetcrime/api)

- Puede encontrar un script de uso sencillo [aquí](https://github.com/adamamer20/streetcrime/blob/main/examples/Milan/Simple%20RAT/model.py)

### Componentes clave
#### City
La clase City representa el espacio geográfico para las interacciones de los agentes, creado utilizando un CRS y una consulta de OpenStreetMap. Cuenta con una red vial avanzada y rutas más cortas precalculadas para mayor eficiencia. Los edificios se categorizan en hogares, lugares de trabajo y actividades potenciales. También se categorizan como abiertos o cerrados durante la noche. Puede encontrar la categorización [aquí](https://github.com/adamamer20/streetcrime/blob/main/src/streetcrime/space/city.py#L360C13-L360C33).

*Nota: Los datos de ciudades grandes pueden tardar más en descargarse; se recomienda comenzar con áreas más pequeñas para las pruebas iniciales.

```python
from streetcrime.space.city import City
from streetcrime.model import StreetCrime
from streetcrime.agents.criminal import Pickpocket, Robber
from streetcrime.agents.police_agent import PoliceAgent
from streetcrime.agents.worker import Worker

milan = City(crs="EPSG:32632", city_name="Milan, Italy")
milan.load_data()
```

#### Agentes
A continuación se presenta un diagrama que muestra los agentes actualmente soportados:

<img src="https://github.com/adamamer20/streetcrime/blob/main/docs/images/classes.png" width="50%"/>

Los nuevos agentes deben crearse a partir de clases existentes. Por ejemplo, un Ladrón de casas podría heredar de la clase Criminal.


##### Clases base
- **Mover**: Esta es la clase fundamental para todos los agentes en movimiento dentro del modelo. La clase Mover está equipada con la capacidad de navegar por la ciudad utilizando su red vial. Los atributos clave incluyen un identificador único, un punto geométrico que representa la ubicación del agente y un estado que indica su actividad actual. La clase también maneja la selección de actividades aleatorias para los agentes y gestiona su movimiento hacia estas ubicaciones. Al llegar a un destino, comienza una cuenta regresiva que determina cuándo el agente puede moverse nuevamente. Las mejoras futuras incluirán reglas diversas para la selección de actividades.

- **InformedMover**: Esta clase representa una versión avanzada de Mover, planificada para una implementación futura. Los agentes InformedMover tendrán acceso a un subconjunto de información global del modelo, como crímenes recientes o áreas de mucho tráfico, basado en un umbral definido. Esta característica permite una toma de decisiones y patrones de movimiento más sofisticados.

- **Resident**: A partir del InformedMover, la clase Resident cuenta con una ubicación de hogar designada y un período de descanso especificado, durante el cual el agente debe permanecer en casa. Esta clase simula los patrones de vida diaria de los habitantes de la ciudad.

##### Clases implementables

- **PoliceAgent**: Una subclase de InformedMover, el PoliceAgent funciona como un guardián dentro de la ciudad. Su papel principal es la prevención del delito; cuando están en estrecha proximidad a un agente Criminal, pueden inhibir actividades criminales, desempeñando así un papel crucial en la representación del modelo de la aplicación de la ley y la dinámica de seguridad pública.

- **Worker**: Derivada de la clase Resident, los Worker son habitantes típicos de la ciudad con rutinas definidas, incluidos lugares de trabajo y horarios. Representan posibles objetivos para los agentes criminales, poseyendo un atributo variable de 'crime_attractiveness' que influye en su probabilidad de ser victimizados. Esta clase ilustra los patrones diarios de los residentes de la ciudad y sus interacciones con otros tipos de agentes.

- **Criminal (Pickpocket y Robber)**: Estos agentes, derivados de la clase Resident, son centrales para la simulación de actividades criminales dentro de la ciudad. Los criminales buscan activamente oportunidades para cometer delitos, seleccionando sus objetivos basándose en oportunidades percibidas y atributos de las víctimas. Las subclases Pickpocket y Robber difieren en sus tácticas operativas y en los criterios para actos criminales exitosos. Por ejemplo, los Pickpocket prosperan en entornos concurridos, mientras que los Robber pueden enfrentar desafíos adicionales, como la posible resistencia de los Worker.

### Modelo
Después de inicializar una ciudad a elección y haber cargado los datos, la ejecución implica inicializar los agentes y ejecutar la simulación. 

```python
model = StreetCrime(milan)
model.create_agents(
    p_agents={Worker: 0.85, PoliceAgent: 0.05, Pickpocket: 0.05, Robber: 0.05}, 
    n_agents=1000
)

#Run the model for 3 days considering time steps of 10 minutes
model.run_model(days=3, len_step=10)
```

Los usuarios también pueden guardar el estado de la simulación para su análisis y reanudar desde estados guardados. Se pueden generar gráficos de crímenes y agentes.

```python
#Save the state of the simulation
model.save_state()

#Plot the model at current state
model.plot()
```

## ¿Qué sigue?
Si encuentra algún desafío al utilizar el paquete, o si tiene sugerencias para mejoras de características, no dude en abrir un problema (issue) en este repositorio de GitHub. 

- Habilitar el acceso a información para InformedMover.
- Incorporar datos de rutas de transporte público.
- Integrar diversas teorías criminológicas (p. ej. ).
- Mejorar las capacidades de visualización, incluyendo la generación de vídeo en vivo o posterior a la simulación.
- Fortalecer la integración con mesa y mesa-frames.
- Introducir varios tipos de criminales (p. ej. Ladrones de casas).
- Implementar ML para la optimización de parámetros.
