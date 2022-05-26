import os
import pickle
import random
import json
import tqdm
import io
import sys
import multiprocessing

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont,ImageChops

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams



def getformuladata(path, nums=500000):

    """
    Get a python list of LaTex formula from the corpus
    """

    with open(path, "r", encoding='utf-8') as f:
        data = f.readlines()
        formulas = []
        count = 0
        for line in data:
            formula = line.strip("\n").split("\t")[-1]
            formula = r"%s"%(formula)
            if len(formula) > 1:
                formulas.append(formula)
                count+=1
            if count >= nums:
                break
    return formulas


background_files = [os.path.join('bg_kept2', item)
                    for item in os.listdir('bg_kept2')]

def create_bg(size):
    """
    randomly choose a background picture from the given dir, randomly crop or resize it

    Args:
        width
        height
    Return:
        resized img
    """

    height, width = size
    if random.random() < 0.4:
        img = np.ones((*size, 3), np.uint8)
        for i in range(3):
            img[:, :, i] *= int(random.uniform(100,255))
        return img


    background_file = random.choice(background_files)
    orig_img = cv2.imread(background_file, cv2.IMREAD_COLOR)

    if orig_img.shape[0] > height and orig_img.shape[1] > width:
        random_x = random.randint(0, orig_img.shape[0] - height)
        random_y = random.randint(0, orig_img.shape[1] - width)
        img = orig_img[random_x:random_x + height, random_y:random_y + width]
    else:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, height, orig_img.shape[0]):
            for j in range(0, width, orig_img.shape[1]):
                for x in range(orig_img.shape[0]):
                    for y in range(orig_img.shape[1]):
                        if i + x >= height or j + y >= width:
                            continue
                        img[i + x, j + y] = orig_img[x, y]
    return img



class Cursor:

    """
    keep track of the current position when creating the fake image
    """

    def __init__(self, bg_size):
        self.bg_size = bg_size

        self.lim_y1, self.lim_y2 = int(self.bg_size[0] * 0.01), int(self.bg_size[0] * 0.95)
        self.lim_x1, self.lim_x2 = int(self.bg_size[1] * 0.01), int(self.bg_size[1] * 0.7)
        self.pos = [random.randint(self.lim_x1, self.lim_x2), random.randint(self.lim_y1, self.lim_y2//6)]  # x, y

    def pos_check(self):
        if self.lim_x1 <= self.pos[0] <= self.lim_x2 and self.lim_y1 <= self.pos[1] <= self.lim_y2:
            return True
        else:
            return False



def get_img_bound(cur_pos, formula_size, bg_size):

    """
    Args:
        cur_pos: position of the cursor
        formula_size: desired size of the formula img
        bg_size: size of the background img
    Return:
        The size of the boundary for the formula img to be cropped
    """

    if cur_pos[0] + formula_size[1] >= bg_size[1]:
        w = bg_size[1] - cur_pos[0]-1
    else:
        w = formula_size[1]
    if cur_pos[1] + formula_size[0] >= bg_size[0]:
        h = bg_size[0] - cur_pos[1]-1
    else:
        h = formula_size[0]
    return (0, 0, w, h)





class RandomChineseGenerator:

    def __init__(self, ch, maxlength=18):
        self.text = ch
        self.len = len(ch)
        self.maxlength = maxlength

    def get_random_text(self, lines=1, txt_length=0):

        """
        grab a sentence from the corpus
        """

        assert txt_length >= 0, "txt_length should be greater than zero"
        if txt_length == 0:
            length = random.randint(1, self.maxlength)
        else:
            length = txt_length
        start = random.randint(0, self.len-length-1)
        assert lines >= 1, "lines < 1:"
        if lines == 1:
            return self.text[start:start+length]
        else:
            ans = ""
            for i in range(lines-1):
                ans += self.text[start:start+length]
                ans += "\n"
                start += length
            ans += self.text[start:start+length-random.randint(0,length-1)]
            return ans


class RandomEnglishGenerator:

    def __init__(self, word_list, maxlength=12):
        self.word_list = word_list
        self.len = len(self.word_list)
        self.maxlength = maxlength

    def get_random_text(self, lines=1, txt_length = 0):

        """
        grab a sentence from the corpus
        """

        assert txt_length >= 0, "txt_length should be greater than zero"
        if txt_length == 0:
            length = random.randint(1, self.maxlength)
        else:
            length = txt_length
        start = random.randint(0, self.len-length-1)
        ans = ""
        if lines == 1:
            for word in self.word_list[start:start+length]:
                ans += (word + " ")
            return ans
        else:
            for i in range(lines-1):
                for word in self.word_list[start:start + length]:
                    ans += (word + " ")
                ans += "\n"
                start += length
            for word in self.word_list[start:start + length-random.randint(0,length-1)]:
                ans += (word + " ")
            return ans


class SingleFormulaFetcher:

    def __init__(self, path):
        self.formula_list = getformuladata(path)


    def get_img(self, fontsize=20, fontfamily="stix", fontcolor=(1, 1, 1)):

        """
        Create an image of formula using the LaTeX corpus
        """

        tex = random.choice(self.formula_list)
        buf = io.BytesIO()
        plt.figure(figsize=(30, 30))
        config = {
            "mathtext.fontset": fontfamily,
        }
        rcParams.update(config)
        plt.axis('off')
        try:
            plt.text(0.05, 0.5, f'${tex}$', size=fontsize)
            plt.savefig(buf, format='jpg')
            plt.close()
            im = Image.open(buf)
            bg = Image.new(im.mode, im.size, (255, 255, 255))
            diff = ImageChops.difference(im, bg)
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()

            buf2 = io.BytesIO()
            plt.figure(figsize=(30, 30))
            config = {
                "mathtext.fontset": fontfamily,
            }
            rcParams.update(config)
            plt.axis('off')
            plt.text(0.05, 0.5, f'${tex}$', size=fontsize, color=fontcolor)
            plt.savefig(buf2, transparent=True)
            plt.close()
            im = Image.open(buf2)
            return im.crop(bbox)
        except:
            return None


class FontLib:

    """
    randomly choose the font size and type from the font library
    """

    def __init__(self, path):

        self.path = path
        self.font_file = [os.path.join(font_path, file) for file in os.listdir(font_path)]
        self.font_dict = {}
        for i in range(len(self.font_file)):
            self.font_dict[i] = {}
            for fontsize in [12, 18, 22, 28, 32, 38, 42]:
                self.font_dict[i][fontsize] = ImageFont.truetype(self.font_file[i], fontsize)


    def get_font(self, fontkey=0, size=20):
        length = len(self.font_file)
        if size < 20:
            fontsize = 18
        elif 20 <= size < 25:
            fontsize = 22
        elif 25 <= size < 30:
            fontsize = 28
        elif 30 <= size < 35:
            fontsize = 32
        elif 35 <= size < 40:
            fontsize = 38
        else:
            fontsize = 42

        fontkey = fontkey % length
        return self.font_dict[fontkey][fontsize]



class TextFormulaDrawer:

    """
    this class draws embedded, isolated formulas and plain text with random size, position and color
    """

    def __init__(self, formulafetcher, ch_generator, eng_generator, fontlib):

        self.formulaFetcher = formulafetcher
        self.ch_generator = ch_generator
        self.eng_generator = eng_generator
        self.fontlib = fontlib

    def draw_embedded(self, draw, bg_img, imgsize, start_pos, fontkey=0, fontsize=20, language="ch", colors=(1.0, 1.0, 1.0), formula_family="stix"):
        formula_img = self.formulaFetcher.get_img(fontsize-11, formula_family, fontcolor=colors)
        totaltry = 3
        while totaltry > 0:
            if formula_img is None or imgsize[1] - formula_img.width - start_pos[0] < 0.3 * imgsize[1]:
                formula_img = self.formulaFetcher.get_img(fontsize-11, formula_family, fontcolor=colors)
                totaltry -= 1
            else:
                break
        if formula_img is None:
            return start_pos[1], None

        blank_size =imgsize[1] - formula_img.width - start_pos[0]

        if blank_size <= 0.3 * imgsize[1]:
            return start_pos[1], None
        else:
            flag = random.choice([1, 2, 2, 2, 2, 2, 3])

            font = self.fontlib.get_font(fontkey, fontsize)
            if language == "ch":
                max_text_length = max(0.6 * blank_size // fontsize, 2)
                text = self.ch_generator.get_random_text(lines=1, txt_length=random.randint(2, max_text_length))
            else:
                max_text_length = max(0.4 * blank_size // fontsize, 2)
                text = self.eng_generator.get_random_text(lines=1, txt_length=random.randint(2, max_text_length))
                width_text, height_text = draw.textsize(text, font)
                if imgsize[1] - width_text - start_pos[0] < 50:
                    text = self.ch_generator.get_random_text(lines=1, txt_length=random.randint(2, max_text_length))
            width_text, height_text = draw.textsize(text, font)
            offset_x, offset_y = font.getoffset(text)
            width_text += offset_x
            height_text += offset_y
            fillColor = (int(255 * colors[0]), int(255 * colors[1]), int(255 * colors[2]))
            if flag == 1 or flag == 2:
                draw.text(tuple(start_pos), text, font=font, fill=fillColor)
                x2 = width_text + start_pos[0] + random.randint(1,10)
                y2 = start_pos[1]
                formula_size = [formula_img.height, formula_img.width]
                img_bound = get_img_bound((x2, y2), formula_size, imgsize)
                bg_img.paste(formula_img.crop(img_bound), (x2,y2), formula_img.crop(img_bound))
            else:
                x2 = start_pos[0]
                y2 = start_pos[1]
                formula_size = [formula_img.height, formula_img.width]
                img_bound = get_img_bound((x2, y2), formula_size, imgsize)
                bg_img.paste(formula_img.crop(img_bound), (x2,y2), formula_img.crop(img_bound))
                x3 = start_pos[0] + img_bound[2] + random.randint(1, 10)
                draw.text((x3, y2), text, font=font, fill=fillColor)

            if flag == 2 and imgsize[0] - x2 - formula_img.width > 40:
                if language == "ch":
                    max_text_length = max(0.6 * blank_size // fontsize, 2)
                    text = self.ch_generator.get_random_text(lines=1, txt_length=random.randint(2, max_text_length))
                else:
                    max_text_length = max(0.4 * blank_size // fontsize, 2)
                    text = self.eng_generator.get_random_text(lines=1, txt_length=random.randint(2, max_text_length))
                draw.text((x2 + img_bound[2] + random.randint(2,10), y2),text, font=font, fill=fillColor)
            return max(y2 + img_bound[3], y2 + height_text), (x2, y2, x2+img_bound[2], y2+img_bound[3])

    def draw_isolated(self, bg_img, start_pos, font_size=20, colors=(1.0, 1.0, 1.0), formula_family="stix"):
        pil_formula_img = self.formulaFetcher.get_img(font_size-11, formula_family, fontcolor=colors)
        while pil_formula_img is None:
            pil_formula_img = self.formulaFetcher.get_img(font_size-11, formula_family, fontcolor=colors)
        formula_size = [pil_formula_img.height, pil_formula_img.width]
        img_bound = get_img_bound(start_pos, formula_size, (bg_img.height,bg_img.width))  # (0, 0, w, h)

        bg_img.paste(pil_formula_img.crop(img_bound), tuple(start_pos), pil_formula_img.crop(img_bound))

        return (start_pos[0], start_pos[1], start_pos[0]+img_bound[2], start_pos[1]+img_bound[3])

    def draw_text(self, draw, imgsize, start_pos, fontkey=0, fontsize=20, language="ch", colors=(1.0, 1.0, 1.0)):
        fillColor = (int(255 * colors[0]), int(255 * colors[1]), int(255 * colors[2]))
        lines = random.choice([1,1,1,1,1,1,1,1,2,2,2])
        font = self.fontlib.get_font(fontkey, fontsize)
        if language == "ch":
            text = self.ch_generator.get_random_text(lines)
        else:
            text = self.eng_generator.get_random_text(lines)
        width_text, height_text = draw.textsize(text, font)
        offset_x, offset_y = font.getoffset(text)
        width_text += offset_x
        height_text += offset_y
        x2 = width_text + start_pos[0]
        y2 = height_text + start_pos[1]
        draw.text(tuple(start_pos), text, font=font, fill=fillColor)

        return ( min(x2,imgsize[1]), min(y2,imgsize[0]) )


class ImageGenrator:

    """
    generate the fake images and its corresponding json annotations, save them in output_path
    """

    def __init__(self, text_formula_drawer, output_path):
        self.output_path = output_path
        self.uni_drawer = text_formula_drawer

    def generate_img(self, k):
        padding_size = 20
        size_list = [(672, 672), (608, 768), (768, 608)]
        rand_color = (random.uniform(0.0, 0.35), random.uniform(0.0, 0.35), random.uniform(0.0, 0.35))
        bg_size = random.choice(size_list)
        bg_img = create_bg(bg_size)
        bg_img = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))

        cursor = Cursor(bg_size)
        draw = ImageDraw.Draw(bg_img)
        language = "ch" if random.random() < 0.75 else "eng"
        font_size = random.randint(25, 45) if language == "ch" else random.randint(25, 40)
        font_key = random.randint(1, 10000)
        formula_font_family = random.choice(['dejavusans', 'dejavuserif', 'cm', 'stix','stixsans'])
        json_dict = {}
        json_dict["result"] = {}
        data = []

        for i in range(20):
            start_pos = tuple(cursor.pos)
            if random.random() < 0.4:
                y, bbox = uni_drawer.draw_embedded(draw, bg_img, bg_size, start_pos, fontkey=font_key,
                                                   fontsize=font_size, language=language, colors=rand_color, formula_family=formula_font_family)
                if bbox:
                    box_info = {"category": ["0"],
                                "coordinate": [
                                    {"x": bbox[0] - 1 + padding_size, "y": bbox[1] - 1 + padding_size},
                                    {"x": bbox[2] + 1 + padding_size, "y": bbox[1] - 1 + padding_size},
                                    {"x": bbox[2] + 1 + padding_size, "y": bbox[3] + 1 + padding_size},
                                    {"x": bbox[0] - 1 + padding_size, "y": bbox[3] + 1 + padding_size},
                                ]}
                    data.append(box_info)
                    y += random.randint(1, 2 * (bbox[3]-bbox[1]))
            else:
                if random.random() < 0.7:
                    bbox = uni_drawer.draw_isolated(bg_img, start_pos, font_size=font_size, colors=rand_color, formula_family=formula_font_family)
                    box_info = {"category": ["1"],
                                "coordinate": [
                                    {"x": bbox[0] - 1 + padding_size, "y": bbox[1] - 1 + padding_size},
                                    {"x": bbox[2] + 1 + padding_size, "y": bbox[1] - 1 + padding_size},
                                    {"x": bbox[2] + 1 + padding_size, "y": bbox[3] + 1 + padding_size},
                                    {"x": bbox[0] - 1 + padding_size, "y": bbox[3] + 1 + padding_size},
                                ]}
                    data.append(box_info)
                    y = bbox[3] + random.randint(1, bbox[3]-bbox[1])
                    x = bbox[2]
                else:
                    end_point = uni_drawer.draw_text(draw, bg_size, start_pos, fontkey=font_key, fontsize=font_size, language=language, colors=rand_color)
                    y = end_point[1] + min(50, random.randint(1, end_point[1] - start_pos[1]))
                    x = end_point[0]

                if random.random() > 0.8 and x < 0.5 * bg_size[1]:
                    start_pos = (random.randint(x+20, int(0.7*bg_size[1])), start_pos[1])
                    bbox = uni_drawer.draw_isolated(bg_img, start_pos, font_size=font_size, colors=rand_color, formula_family=formula_font_family)
                    box_info = {"category": ["1"],
                                "coordinate": [
                                    {"x": bbox[0] - 1 + padding_size, "y": bbox[1] - 1 + padding_size},
                                    {"x": bbox[2] + 1 + padding_size, "y": bbox[1] - 1 + padding_size},
                                    {"x": bbox[2] + 1 + padding_size, "y": bbox[3] + 1 + padding_size},
                                    {"x": bbox[0] - 1 + padding_size, "y": bbox[3] + 1 + padding_size},
                                ]}
                    data.append(box_info)
                    y = max(bbox[3] + random.randint(1, 2 * (bbox[3] - bbox[1])), y)
            cursor.pos = [random.randint(10, int(0.2 * bg_size[1])), y]
            if not cursor.pos_check():
                break

        json_dict["result"]["data"] = data
        json_str = json.dumps(json_dict)

        img_OpenCV = cv2.cvtColor(np.asarray(bg_img), cv2.COLOR_RGB2BGR)
        height, width = bg_size
        if random.random() < 0.5:
            factor = random.uniform(0.7, 0.95)
            img_OpenCV = cv2.resize(img_OpenCV, (int(width * factor), int(height * factor)))
            img_OpenCV = cv2.resize(img_OpenCV, (width, height))
        img_OpenCV = cv2.copyMakeBorder(img_OpenCV, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=[114, 114, 114])


        num = str(k)
        zeros = 6 - len(num)

        img_name = self.output_path + "formulas_with_words_" + "0" * zeros + num + ".png"
        json_name = self.output_path + "formulas_with_words_" + "0" * zeros + num + ".json"

        with open(json_name, 'w') as json_file:
            json_file.write(json_str)
        cv2.imwrite(img_name, img_OpenCV)



if __name__ == "__main__":

    num = int(sys.argv[1])
    output_path = str(sys.argv[2])
    num_process = int(sys.argv[3])

    with open("word4gen/english_type_list.pkl", "rb") as f1:
        english_wordlist = pickle.load(f1)
    with open("word4gen/chinese_type_string.pkl", "rb") as f2:
        chinese_string = pickle.load(f2)

    chineseGen = RandomChineseGenerator(chinese_string)
    englishGen = RandomEnglishGenerator(english_wordlist)
    formulaFetcher = SingleFormulaFetcher("word4gen/latex_formula_list.txt")

    font_path = "font/"
    fontlib = FontLib(font_path)

    os.makedirs(output_path, exist_ok=True)
    uni_drawer = TextFormulaDrawer(formulaFetcher, chineseGen, englishGen, fontlib)
    image_genrator = ImageGenrator(uni_drawer, output_path)

    input_list = list(range(num))
    p = multiprocessing.Pool(num_process)
    res = list(tqdm.tqdm(p.imap(image_genrator.generate_img, input_list), total=len(input_list)))
    p.close()
    p.join()
