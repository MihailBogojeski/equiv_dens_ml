// Client side C/C++ program to demonstrate Socket programming
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include "json/json.h"
#include "json/reader.h"
#include "json/writer.h"
#include "json/value.h"
#define PORT 50007
   
int main(int argc, char const *argv[])
{
    int sock = 0;
    struct sockaddr_in serv_addr;

    std::ifstream in_file("/home/mihail/Documents/workspace/schnet-tfn/equiv_dens/datasets/ethanol_10_rot.xyz");
    std::string line;
    int line_count = 0;
    std::string word;
    char delim = ' ';
    Json::Value mol_dict;
    Json::Value positions;
    Json::Value atom_types;
    Json::StreamWriterBuilder builder;

    int n_atoms = 9;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        printf("\nConnection Failed \n");
        return -1;
    }
    // for(int j = 0; j < 10; j++){
    bool not_EOF = 1;
    while(not_EOF){
      not_EOF = 0;
      Json::Value mol_pos;
      while(std::getline(in_file, line)){
          int word_count = 0;
          Json::Value atom_pos;
          if((line_count % (n_atoms + 2)) < 2){
              line_count++;
              continue;
          } 
          std::cout << "line count: " << line_count << std::endl;
          std::cout << "line: " << line << std::endl;
          std::istringstream ss(line);
          while(std::getline(ss, word, delim)){
              if(word == ""){
                  continue;
              }
              std::cout << "word: " << word << std::endl;
              std::cout << "word count: " << word_count << std::endl;
              if(word_count == 0){
                  if(line_count < (n_atoms + 2)){
                      atom_types.append(word);
                  }
              }
              else{
                  atom_pos.append(word);
              }
              word_count++;
          }
          line_count++;
          mol_pos.append(atom_pos);
          if((line_count % (n_atoms + 2)) == 0){
              positions.append(mol_pos);
              not_EOF = 1;
              break;
          } 
      }
    }
    mol_dict["positions"] = positions;
    mol_dict["atom_types"] = atom_types;
    std::string output = Json::writeString(builder, mol_dict);
    std::cout << output << std::endl;
    in_file.close();
    //
    // char *hello = "Hello from client";
    char buffer[1024] = {0};
    //
    // for(int i=0; i<3; i++){
    std::cout << "Output length: " << output.length() << std::endl;
    std::string output_str = std::to_string(output.length());
    send(sock, output_str.c_str(), output_str.length() , 0);
    std::cout << "Sent output length" << std::endl;
    read(sock, buffer, 1024);
    std::cout << "server response: " << buffer << std::endl; 
    send(sock , output.c_str(), output.length() , 0);
    std::cout << "Atomic positions sent!" << std::endl;
    memset(buffer, 0, sizeof(buffer));
    read(sock, buffer, 64);
    std::cout << "server sent length: " << buffer << std::endl; 
    int recv_len = atoi(buffer);
    std::string output_str2 = "Received length " + std::to_string(recv_len);
    std::cout << "Sending following message to server: " << output_str2 << std::endl;
    send(sock , output_str2.c_str(), output_str2.length() , 0);
    char json_in_buffer[recv_len] = {0};
    read(sock, json_in_buffer, recv_len);
    std::cout << "Server json response" << std::endl;
    std::cout << json_in_buffer << std::endl;
    // }
    char exit_buffer[1024] = {0};
    std::string exit_str = "exit";
    std::string exit_len = std::to_string(exit_str.length());
    // std::string exit_len = "Sending some random message";
    std::cout << "Exit msg length: " << exit_len << std::endl;
    std::cout << "Len msg len" << exit_len.length() << std::endl;
    send(sock, exit_len.c_str(), exit_len.length(), 0);
    std::cout << "Sent exit length" << std::endl;
    read(sock, exit_buffer, 1024);
    std::cout << "server response: " << exit_buffer << std::endl; 
    send(sock , exit_str.c_str(), exit_str.length() , 0);
    std::cout << "Exit command sent!" << std::endl;
    close(sock);
    return 0;
}
