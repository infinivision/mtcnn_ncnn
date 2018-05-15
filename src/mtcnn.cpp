#include <algorithm>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "mtcnn.h"
#include "utils.h"

bool cmpScore(orderScore lsh, orderScore rsh){
    if(lsh.score<rsh.score)
        return true;
    else
        return false;
}


MTCNN::MTCNN(){

}

MTCNN::MTCNN(const std::string& model_path){
    std::vector<std::string> param_files = {
		model_path+"/det1.param",
		model_path+"/det2.param",
		model_path+"/det3.param"
	};

	std::vector<std::string> bin_files = {
		model_path+"/det1.bin",
		model_path+"/det2.bin",
		model_path+"/det3.bin"
	};

	pnet_.load_param(param_files[0].c_str());
	pnet_.load_model(bin_files[0].c_str());
	rnet_.load_param(param_files[1].c_str());
	rnet_.load_model(bin_files[1].c_str());
	onet_.load_param(param_files[2].c_str());
	onet_.load_model(bin_files[2].c_str());
    // pnet_.load_param("det1.param");
    // pnet_.load_model("det1.bin");
    // rnet_.load_param("det2.param");
    // rnet_.load_model("det2.bin");
    // onet_.load_param("det3.param");
    // onet_.load_model("det3.bin");
}



void MTCNN::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, std::vector<orderScore>& bboxScore_, float scale){
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    std::cout << "=====generate Bbox====" << std::endl; 
    std::cout << "scale: " << scale << " score width: " << score.w << "\theigth: " << score.h << std::endl;
    std::cout << "score channel 1 total: " << score.channel(1).total() << std::endl;
    std::cout << "location channel num: " << location.c << "\twidth: " << location.w << "\theigth: " << location.h << std::endl;
    //score p
    float *p = score.channel(1);
    Bbox bbox;
    orderScore order;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*col+1)/scale);
                bbox.y1 = round((stride*row+1)/scale);
                bbox.x2 = round((stride*col+1+cellsize)/scale);
                bbox.y2 = round((stride*row+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=location.channel(channel)[0];
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            // plocal++;
        }
    }
}

// nms: non maximum suppression
// IOU: intersection over union
void MTCNN::nms(std::vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty()){
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while(bboxScore_.size()>0){
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if(order<0)continue;
        if(boundingBox_.at(order).exist == false) continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it

        for(int num=0;num<boundingBox_.size();num++){
            if(boundingBox_.at(num).exist){
                // Compute ordinates of intersection-over-union(IOU)
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?boundingBox_.at(num).x1:boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?boundingBox_.at(num).y1:boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?boundingBox_.at(num).x2:boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?boundingBox_.at(num).y2:boundingBox_.at(order).y2;

                //Compute areas of intersection-over-union
                maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
                maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
                IOU = maxX * maxY;

                if(!modelname.compare("Union"))
                    IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if(!modelname.compare("Min")){
                    IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
                }
                if(IOU>overlap_threshold){
                    boundingBox_.at(num).exist=false;
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
                        if((*it).oriOrder == num) {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for(unsigned int i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;
}

void MTCNN::refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }

    std::cout << "====refine and square box====" << std::endl;
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        if((*it).exist){
            bbw = (*it).x2 - (*it).x1 + 1;
            bbh = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
            y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
            x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
            y2 = (*it).y2 + (*it).regreCoord[3]*bbh;

            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if((*it).x1<0) {
                (*it).x1=0;
            }

            if((*it).y1<0) {
                (*it).y1=0;
            }

            if((*it).x2>width) {
                (*it).x2 = width - 1;
            }

            if((*it).y2>height) {
                (*it).y2 = height - 1;
            }

            std::cout << "x1: " << it->x1 << "\tx2: " << it->x2 << "\ty1: " << it->y1 << "\ty2: " << it->y2
             << "\t(x2-x1)= " << it->x2 - it->x1 << "\t(y2-y1)= " << it->y2 - it->y1 << std::endl;
            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}

void MTCNN::detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox_){
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();

    img = img_;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);

    std::cout << "image height: " << img_h << "\timage width: " << img_w << std::endl;
    float minl = img_w<img_h?img_w:img_h;
    int MIN_DET_SIZE = 12;
    int minsize = 80;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }

    for(std::vector<float>::iterator it=scales_.begin(); it!=scales_.end(); it++) {
        std::cout << *it << std::endl;
    }

    orderScore order;
    int count = 0;

    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        ncnn::Mat in;
        resize_bilinear(img_, in, ws, hs);

        ncnn::Extractor ex = pnet_.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        
        std::vector<Bbox> boundingBox_;
        std::vector<orderScore> bboxScore_;
        generateBbox(score_, location_, boundingBox_, bboxScore_, scales_[i]);
        nms(boundingBox_, bboxScore_, nms_threshold[0]);

        for(vector<Bbox>::iterator it=boundingBox_.begin(); it!=boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        bboxScore_.clear();
        boundingBox_.clear();
    }
    
    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, img_h, img_w);
    std::cout << "firstBbox_.size() = " << firstBbox_.size() << std::endl;

    //second stage
    count = 0;
    for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;            
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);

            ncnn::Mat in;
            resize_bilinear(tempIm, in, 24, 24);

            ncnn::Extractor ex = rnet_.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox;
            ex.extract("prob1", score);
            ex.extract("conv5-2", bbox);

            if(score.channel(1)[0]>threshold[1]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox.channel(channel)[0];//*(bbox.data+channel*bbox.cstep);
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];//*(score.data+score.cstep);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    std::cout << "secondBbox_.size() = " << secondBbox_.size() << std::endl;    
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, img_h, img_w);

    //third stage 
    count = 0;
    for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 48, 48);
            ncnn::Extractor ex = onet_.create_extractor();
            ex.set_light_mode(true);

            ex.input("data", in);
            ncnn::Mat score, bbox, keyPoint;
            ex.extract("prob1", score);
            ex.extract("conv6-2", bbox);
            ex.extract("conv6-3", keyPoint);      
            if(score.channel(1)[0]>threshold[2]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox.channel(channel)[0];
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint.channel(num)[0];
                    (it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1)*keyPoint.channel(num+5)[0];
                }

                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else
                (*it).exist=false;
            }
        }

    std::cout << "thirdBbox_.size() = " << thirdBbox_.size() << std::endl;
    if(count < 1)
        return;
    refineAndSquareBbox(thirdBbox_, img_h, img_w);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
    finalBbox_ = thirdBbox_;
}

