# NetRail-Incident-Analysis Package

NetRail-Incident-Analysis is a prototype refactoring of a older rail incident-delay model, based on data from a range of sources, but mostly Network Rail. This model has been redesigned for reanalysis with increased granularity and modularity. 

At the current state, the public repo contains `rail_io`, subpackage containing code for importing from a range of sources. 
## src/Rail_data/Rail_io
This subpackage provides a prototype foundation for importing and extending rail delay analytics workflows—particularly useful if you’re working with similar models or need some examples of wrangling network rail data, handling data formats,`CIF` decoding etc.
