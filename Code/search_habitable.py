import math

# Calculate the radius of a star using the Stefan-Boltzmann law.
def calculate_star_radius(luminosity, temperature):
    stefan_boltzmann_constant = 5.670374419e-8  # W/m^2/K^4
    radius = math.sqrt(luminosity / (4 * math.pi * stefan_boltzmann_constant * temperature**4))
    return radius

# Calculate the inner and outer boundaries of the habitable zone in AU.
def calculate_habitable_zone(luminosity):
    inner_boundary = math.sqrt(luminosity / 1.1)
    outer_boundary = math.sqrt(luminosity / 0.53)
    return inner_boundary, outer_boundary

# Calculate the orbital distance of a planet given the stellar flux and luminosity.
# Only valid for orbits with low eccentricity.
def calculate_orbital_distance(flux, luminosity):
    distance = math.sqrt(luminosity / (4 * math.pi * flux))
    return distance

# Calculate the mass of a star using the mass-luminosity relation.
# Only valid for main-sequence stars.
def calculate_stellar_mass(luminosity):
    sun_mass = 1.989e30  # kg
    sun_luminosity = 3.828e26  # W
    return math.pow(luminosity / sun_luminosity, 1/3.5) * sun_mass

# Calculate the orbital distance of a planet using Kepler's third law.
def calculate_orbital_distance(orbital_period, stellar_mass):
    gravitational_constant = 6.67430e-11  # m^3/kg/s^2
    period = math.cbrt((orbital_period**2 * gravitational_constant * stellar_mass) / (4 * math.pi**2))
    return period