import cv2
import numpy
import functools
import operator
import itertools
import random
import copy
from PIL import Image, ImageDraw

img_file_name = 'mona_lisa_crop.jpg'

population_size = 10
num_of_polygons = 100
num_of_vertices = 6
num_of_mating_parents = 5
mutation_percentage = 1
max_generation = 1000000
shape = (256, 256, 3)


# resize an image
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


# convert image to chromosome
def img2chromosome(img_arr):
    chromosome = numpy.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))
    return chromosome


# convert chromosome to iamge
def chromosome2img(chromosome, img_shape):
    img_arr = numpy.reshape(a=chromosome, newshape=img_shape)
    return img_arr


def generate_rand_polygon():
    rand_points = numpy.int32(numpy.random.random((num_of_vertices, 2)) * shape[0])
    rand_points = list(rand_points.flatten())
    color = random.sample(range(0, 255), 4)
    color[3] = 30
    polygon = (rand_points, color)
    return polygon


def draw_polygons(indv, img_shape):
    image = Image.new('RGB', (shape[0], shape[1]))
    drw = ImageDraw.Draw(image, 'RGBA')
    for i in range(indv.shape[0]):
        drw.polygon(indv[i][0], tuple(indv[i][1]))
    image = numpy.array(image).astype(numpy.uint8)
    return image


# generate the initial population
def initial_population(img_shape, n_individuals):
    init_population = numpy.empty(shape=(n_individuals, num_of_polygons),
                                  dtype=object)
    for indv_num in range(n_individuals):
        # Randomly generating initial population chromosomes genes values.
        init_population[indv_num] = [generate_rand_polygon() for i in init_population[indv_num]]
    return init_population


# get the fitness of an individual
def fitness_fun(target_chrom, indiv_chrom):
    quality = numpy.sum(numpy.abs(target_chrom - indiv_chrom))
    quality = 100 * (1 - quality / (255 * shape[0] * shape[1] * 3))
    return quality


# ge the fitness of all the individuals in the population
def population_fitness(target_chrom, pop):
    fitness = numpy.zeros((population_size,), [('ind_fitness', 'float64'), ('ind_index', 'int')])
    for indv_num in range(population_size):
        pop_indv = img2chromosome(draw_polygons(pop[indv_num], shape))
        fitness[indv_num] = (fitness_fun(target_chrom, pop_indv), indv_num)
    return fitness


# select the fittest individuals to be parents
def selection(pop, fitness, num_of_parents):
    parents = numpy.zeros((num_of_parents, num_of_polygons), dtype=object)
    for i in range(num_of_parents):
        parents[i] = pop[fitness[population_size - i - 1][1]]
    return parents


# generate children from parents
def crossover(parents, img_shape, n_individuals):
    new_population = numpy.empty(shape=(n_individuals, num_of_polygons), dtype=object)

    new_population[0:parents.shape[0], :] = copy.deepcopy(parents)
    num_newly_generated = n_individuals - parents.shape[0]
    parents_permutations = list(itertools.permutations(iterable=numpy.arange(0, parents.shape[0]), r=2))
    selected_permutations = random.sample(range(len(parents_permutations)),
                                          num_newly_generated)
    ref_idx = parents.shape[0]
    for comb in range(len(selected_permutations)):
        selected_comb_idx = selected_permutations[comb]
        selected_comb = parents_permutations[selected_comb_idx]
        half_size = numpy.int32(new_population.shape[1] / 2)
        new_population[ref_idx + comb, 0:half_size] = copy.deepcopy(parents[selected_comb[0],
                                                                     0:half_size])
        new_population[ref_idx + comb, half_size:] = copy.deepcopy(parents[selected_comb[1],
                                                                    half_size:])
    return new_population


# mutation
def mutation(pop, num_parents_mating, mut_percent):
    for idx in range(num_parents_mating, pop.shape[0]):

        rand_idx = numpy.uint32(numpy.random.random(size=numpy.uint32(mut_percent / 100 * pop.shape[1])) * pop.shape[1])

        for i in rand_idx:
            rand_point = random.randint(0, len(pop[idx][i][0]) - 1)
            rand_channel = random.randint(0, 2)
            mutation_type = random.randint(0, 2)

            # select the mutation type
            if mutation_type == 0 or mutation_type == 2:
                pop[idx][i][0][rand_point] = min(shape[0], numpy.abs(numpy.random.normal(
                    pop[idx][i][0][rand_point], 70, 1)))
            elif mutation_type == 1:
                pop[idx][i][1][rand_channel] = random.randint(0, 255)
    return pop


def make_art():
    img = cv2.imread(img_file_name, 1)
    img = image_resize(img, shape[0])
    optimal = img2chromosome(img)

    new_generation = initial_population(img.shape, population_size)

    for generation_num in range(max_generation):

        # get the population fitness
        generation_quality = population_fitness(optimal, new_generation)

        # sort the fitness to get ready for selection
        generation_quality.sort()

        print("Quality of the generation #", generation_num, "=", generation_quality[population_size - 1][0], "%")

        # select the parents
        parents = selection(new_generation, generation_quality, num_of_mating_parents)

        # generate children
        new_generation = crossover(parents, img.shape, population_size)

        # mutate children
        new_generation = mutation(new_generation, num_of_mating_parents, mutation_percentage)

        # display the fittest individual of the population
        img = draw_polygons(parents[0], img.shape)

        name = img_file_name + str(generation_num) + ".jpg"
        if generation_num % 500 == 0:
            cv2.imwrite(name, img)
            cv2.waitKey(30)

        cv2.imshow("image", img)
        cv2.waitKey(1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


make_art()
